#pragma once
// Stage 5: device MMU — page table + per-engine TLB.
//
// Device addresses in sg_cmd are virtual. Engines translate through a
// modeled TLB; a miss walks the page table. sgMalloc installs an
// always-resident identity map (VA == PA) for its range. Managed pages
// are present but host-resident until a fault migrates them into VRAM.
//
// Page-table entries and TLB tags are atomics: the driver writes them on
// map/unmap, the engine threads read them on every access. No lock on the
// translate path.

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>

#include "softgpu/sg_ioctl.h"

namespace softgpu::device {

inline constexpr uint64_t SG_PAGE_SHIFT = 12;
inline constexpr uint32_t SG_TLB_ENTRIES = 64;
static_assert(SG_PAGE_SIZE == (1u << SG_PAGE_SHIFT), "SG_PAGE_SIZE must match the shift");

// PTE: flags in the low 12 bits, physical frame number above that.
// A host pointer does not fit in those bits; host-resident pages keep it
// in a side array.
inline constexpr uint64_t PTE_PRESENT = 1ull << 0;
inline constexpr uint64_t PTE_IN_VRAM = 1ull << 1;
inline constexpr uint64_t PTE_DIRTY = 1ull << 2;
inline constexpr uint64_t PTE_ACCESSED = 1ull << 3;
inline constexpr uint64_t PTE_FRAME_SHIFT = 12;

inline uint64_t page_floor(uint64_t a) { return a & ~(SG_PAGE_SIZE - 1); }
inline uint64_t page_offset(uint64_t a) { return a & (SG_PAGE_SIZE - 1); }
inline uint64_t pages_for(uint64_t addr, uint64_t len) {
    if (len == 0) return 0;
    return (page_floor(addr + len - 1) - page_floor(addr)) / SG_PAGE_SIZE + 1;
}
inline uint64_t pte_pa(uint64_t pte) { return (pte >> PTE_FRAME_SHIFT) << SG_PAGE_SHIFT; }

struct Translate {
    enum Status { Ok, NotPresent, HostResident } status = NotPresent;
    uint64_t pa = 0; // VRAM byte offset of the page base (status == Ok)
};

// Snapshot of one page, no TLB update. The fault thread uses this; engines
// use translate(), which may fill a TLB.
struct PageInfo {
    enum Kind { Absent, Host, Vram } kind = Absent;
    uint64_t host = 0; // kind == Host
    uint64_t pa = 0;   // kind == Vram
};

class Mmu {
public:
    explicit Mmu(uint64_t va_bytes);

    // Install / remove VRAM mappings. `va` and `pa` must be page-aligned;
    // `npages` pages are mapped contiguously. Safe to call while engines
    // run: unmap drops the PTE, bumps a generation, and shoots the TLB
    // down. The driver must still not recycle the frame until in-flight
    // commands have retired (same deferred-free rule as VRAM itself).
    int map_vram(uint64_t va, uint64_t pa, uint64_t npages);
    // Present, not in VRAM. `host` is the address of the first page; the
    // following pages are contiguous. Device access faults instead of
    // returning a pointer.
    int map_host(uint64_t va, uint64_t host, uint64_t npages);
    // Host-resident page becomes the identity VRAM frame at `pa`.
    // Already-resident is success.
    int make_resident(uint64_t va, uint64_t pa);
    int unmap(uint64_t va, uint64_t npages);

    PageInfo page(uint64_t va) const;

    // Translate one VA to a VRAM page base. Updates accessed/dirty and the
    // calling engine's TLB. `engine` selects that TLB; only that engine's
    // thread may call this.
    Translate translate(uint32_t engine, uint64_t va, bool write);

    void reset_stats();

    uint64_t tlb_hits(uint32_t engine) const {
        return tlb_hits_[engine].load(std::memory_order_relaxed);
    }
    uint64_t tlb_misses(uint32_t engine) const {
        return tlb_misses_[engine].load(std::memory_order_relaxed);
    }

private:
    struct TlbEntry {
        std::atomic<uint64_t> vpn{~0ull}; // ~0 = empty
        std::atomic<uint64_t> pte{0};
        uint64_t last_used = 0; // engine thread only; not read by the driver
    };
    // One engine's TLB on its own line so probes do not false-share.
    struct alignas(64) EngineTlb {
        TlbEntry entry[SG_TLB_ENTRIES];
        uint64_t clock = 1; // engine thread only
    };

    Translate walk(uint32_t engine, uint64_t va, bool write);
    void tlb_insert(uint32_t engine, uint64_t vpn, uint64_t pte);
    void tlb_invalidate_vpn(uint64_t vpn);

    std::unique_ptr<std::atomic<uint64_t>[]> pte_;  // index = VPN
    std::unique_ptr<std::atomic<uint64_t>[]> host_; // valid when present and not in VRAM
    uint64_t npages_;
    EngineTlb tlb_[SG_MAX_ENGINES];
    // Bumped after a PTE is cleared so an insert that raced the unmap
    // drops its tag instead of publishing a stale translation.
    std::atomic<uint64_t> tlb_gen_{0};
    std::atomic<uint64_t> tlb_hits_[SG_MAX_ENGINES]{};
    std::atomic<uint64_t> tlb_misses_[SG_MAX_ENGINES]{};
    // map/unmap only. Translate does not take it.
    std::mutex map_lock_;
};

} // namespace softgpu::device
