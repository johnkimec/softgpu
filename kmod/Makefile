# softgpu char device. Build on the VM, against that kernel's headers:
#   make -C /lib/modules/$(uname -r)/build M=$PWD
#   sudo insmod softgpu.ko
obj-m += softgpu.o

KDIR ?= /lib/modules/$(shell uname -r)/build
ccflags-y += -I$(src)/../include

all:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean
