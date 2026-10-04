// SPDX-License-Identifier: GPL-2.0
/*
 * /dev/softgpu — command rings, a doorbell page, and the host interrupt line.
 *
 * The device model stays a userspace thread (it dereferences host pointers,
 * runs userfaultfd, and executes GEMM). This module is the boundary that
 * thread shares with the driver:
 *
 *   SG_IOC_KMOD_MAP    allocate the rings and the doorbell/IRQ page
 *   mmap               map them into the process. Producers store commands
 *                      and ring a doorbell word; there is no ioctl per command.
 *   SG_IOC_KMOD_IRQ    top half. A workqueue item is the threaded ISR: it
 *                      publishes irq_seq and wake_up()s the wait queue.
 *   SG_IOC_KMOD_SLEEP  host waiters sleep there instead of on a futex.
 *
 * The classic SG_IOC_* ordinals are recognized and rejected. Allocation,
 * submission, and fault handling stay in the userspace driver; submission
 * does not enter the kernel at all.
 */

#include <linux/fs.h>
#include <linux/jiffies.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/vmalloc.h>
#include <linux/wait.h>
#include <linux/workqueue.h>

#include "softgpu/sg_ioctl.h"

struct sg_file {
	struct mutex lock;
	wait_queue_head_t wq;
	struct work_struct irq_work;
	void *mem;
	struct sg_kmod_ctrl *ctrl;
	size_t bytes;
};

static void sg_irq_work(struct work_struct *work)
{
	struct sg_file *s = container_of(work, struct sg_file, irq_work);
	u32 seq;

	if (!s->ctrl)
		return;
	seq = smp_load_acquire(&s->ctrl->irq_seq);
	smp_store_release(&s->ctrl->irq_seq, seq + 1);
	wake_up_all(&s->wq);
}

static int sg_open(struct inode *ino, struct file *filp)
{
	struct sg_file *s;

	s = kzalloc(sizeof(*s), GFP_KERNEL);
	if (!s)
		return -ENOMEM;
	mutex_init(&s->lock);
	init_waitqueue_head(&s->wq);
	INIT_WORK(&s->irq_work, sg_irq_work);
	filp->private_data = s;
	return 0;
}

static int sg_release(struct inode *ino, struct file *filp)
{
	struct sg_file *s = filp->private_data;

	cancel_work_sync(&s->irq_work);
	vfree(s->mem);
	kfree(s);
	return 0;
}

static int sg_mmap(struct file *filp, struct vm_area_struct *vma)
{
	struct sg_file *s = filp->private_data;

	if (!s->mem)
		return -ENODEV;
	if ((vma->vm_end - vma->vm_start) > s->bytes)
		return -EINVAL;
	/* pgoff 0: the whole mapping, control page then rings. */
	return remap_vmalloc_range(vma, s->mem, 0);
}

static long sg_do_map(struct sg_file *s, unsigned long arg)
{
	struct sg_kmod_map m;
	void *mem;
	size_t bytes;

	if (copy_from_user(&m, (void __user *)arg, sizeof(m)))
		return -EFAULT;
	if (s->mem)
		return -EBUSY;
	if (m.num_engines == 0 || m.num_engines > SG_MAX_ENGINES ||
	    m.num_channels == 0 || m.num_channels > SG_MAX_CHANNELS)
		return -EINVAL;
	if (m.depth < 2 || m.depth > 65536 || (m.depth & (m.depth - 1)))
		return -EINVAL;

	bytes = sg_kmod_map_bytes(m.num_engines, m.num_channels, m.depth);
	mem = vmalloc_user(bytes);
	if (!mem)
		return -ENOMEM;
	s->mem = mem;
	s->ctrl = mem;
	s->bytes = bytes;
	m.bytes = bytes;
	if (copy_to_user((void __user *)arg, &m, sizeof(m))) {
		vfree(mem);
		s->mem = NULL;
		s->ctrl = NULL;
		s->bytes = 0;
		return -EFAULT;
	}
	return 0;
}

static long sg_do_sleep(struct sg_file *s, unsigned long arg)
{
	struct sg_kmod_sleep sl;
	unsigned long timeout;
	long ret;

	if (copy_from_user(&sl, (void __user *)arg, sizeof(sl)))
		return -EFAULT;
	if (!s->ctrl)
		return -ENODEV;
	timeout = nsecs_to_jiffies(sl.timeout_ns);
	if (timeout == 0)
		timeout = 1;
	ret = wait_event_interruptible_timeout(s->wq,
		smp_load_acquire(&s->ctrl->irq_seq) != sl.seen, timeout);
	if (ret == -ERESTARTSYS)
		return -EINTR;
	return 0;
}

static long sg_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
	struct sg_file *s = filp->private_data;
	long rc;

	switch (cmd) {
	case SG_IOC_KMOD_MAP:
		mutex_lock(&s->lock);
		rc = sg_do_map(s, arg);
		mutex_unlock(&s->lock);
		return rc;
	case SG_IOC_KMOD_IRQ:
		if (!s->ctrl)
			return -ENODEV;
		schedule_work(&s->irq_work);
		return 0;
	case SG_IOC_KMOD_SLEEP:
		return sg_do_sleep(s, arg);
	case SG_IOC_QUERY:
	case SG_IOC_ALLOC:
	case SG_IOC_FREE:
	case SG_IOC_SUBMIT:
	case SG_IOC_STATS:
	case SG_IOC_RESET_STATS:
	case SG_IOC_WAIT:
	case SG_IOC_PIN:
	case SG_IOC_UNPIN:
	case SG_IOC_ALLOC_MANAGED:
	case SG_IOC_UFFD_HOLD:
	case SG_IOC_SET_PRIORITY:
		/* Userspace driver. SUBMIT never gets here: it stores into
		 * the mapping. */
		return -ENOTTY;
	default:
		return -ENOTTY;
	}
}

static const struct file_operations sg_fops = {
	.owner = THIS_MODULE,
	.open = sg_open,
	.release = sg_release,
	.mmap = sg_mmap,
	.unlocked_ioctl = sg_ioctl,
	.llseek = no_llseek,
};

static struct miscdevice sg_misc = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "softgpu",
	.fops = &sg_fops,
	.mode = 0666,
};

static int __init sg_mod_init(void)
{
	return misc_register(&sg_misc);
}

static void __exit sg_mod_exit(void)
{
	misc_deregister(&sg_misc);
}

module_init(sg_mod_init);
module_exit(sg_mod_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("softgpu command rings, doorbell page, and interrupt line");
MODULE_AUTHOR("softgpu");
