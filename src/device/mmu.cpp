#include "device/mmu.h"

#include <cerrno>

namespace softgpu::device {

Mmu::Mmu(uint64_t va_bytes, uint64_t vram_bytes)
    : npages_(va_bytes / SG_PAGE_SIZE), vram_pages_(vram_bytes / SG_PAGE_SIZE) {
    pte_.reset(new std::atomic<uint64_t>[npages_]());
    host_.reset(new std::atomic<uint64_t>[npages_]());
}

int Mmu::map_vram(uint64_t va, uint64_t pa, uint64_t npages) {
    if ((va | pa) & (SG_PAGE_SIZE - 1)) return -EINVAL;
    if (npages == 0) return -EINVAL;
    const uint64_t vpn0 = va >> SG_PAGE_SHIFT;
    const uint64_t pfn0 = pa >> SG_PAGE_SHIFT;
    if (vpn0 >= npages_ || npages > npages_ - vpn0) return -EINVAL;
    if (pfn0 >= vram_pages_ || npages > vram_pages_ - pfn0) return -EINVAL;

    std::lock_guard<std::mutex> g(map_lock_);
    for (uint64_t i = 0; i < npages; ++i)
        if (pte_[vpn0 + i].load(std::memory_order_relaxed) & PTE_PRESENT) return -EEXIST;
    for (uint64_t i = 0; i < npages; ++i) {
        const uint64_t pte = PTE_PRESENT | PTE_IN_VRAM | ((pfn0 + i) << PTE_FRAME_SHIFT);
        pte_[vpn0 + i].store(pte, std::memory_order_release);
        tlb_invalidate_vpn(vpn0 + i);
    }
    return 0;
}

int Mmu::map_host(uint64_t va, uint64_t host, uint64_t npages) {
    if ((va | host) & (SG_PAGE_SIZE - 1)) return -EINVAL;
    if (npages == 0) return -EINVAL;
    const uint64_t vpn0 = va >> SG_PAGE_SHIFT;
    if (vpn0 >= npages_ || npages > npages_ - vpn0) return -EINVAL;

    std::lock_guard<std::mutex> g(map_lock_);
    for (uint64_t i = 0; i < npages; ++i)
        if (pte_[vpn0 + i].load(std::memory_order_relaxed) & PTE_PRESENT) return -EEXIST;
    for (uint64_t i = 0; i < npages; ++i) {
        host_[vpn0 + i].store(host + i * SG_PAGE_SIZE, std::memory_order_relaxed);
        pte_[vpn0 + i].store(PTE_PRESENT, std::memory_order_release);
        tlb_invalidate_vpn(vpn0 + i);
    }
    return 0;
}

int Mmu::make_resident(uint64_t va, uint64_t pa) {
    if ((va | pa) & (SG_PAGE_SIZE - 1)) return -EINVAL;
    const uint64_t vpn = va >> SG_PAGE_SHIFT;
    const uint64_t pfn = pa >> SG_PAGE_SHIFT;
    if (vpn >= npages_ || pfn >= vram_pages_) return -EINVAL;

    std::lock_guard<std::mutex> g(map_lock_);
    const uint64_t cur = pte_[vpn].load(std::memory_order_acquire);
    if (!(cur & PTE_PRESENT)) return -EINVAL;
    if (cur & PTE_IN_VRAM) return 0;
    // Clean: the frame now matches the host backing. ACCESSED gives the
    // page one CLOCK pass before it can be chosen as a victim.
    const uint64_t neu = PTE_PRESENT | PTE_IN_VRAM | PTE_ACCESSED | (pfn << PTE_FRAME_SHIFT);
    pte_[vpn].store(neu, std::memory_order_release);
    // Same protocol as unmap: an insert that sampled the host PTE drops
    // its tag instead of caching a translation that is no longer host.
    tlb_gen_.fetch_add(1, std::memory_order_seq_cst);
    tlb_invalidate_vpn(vpn);
    return 0;
}

uint64_t Mmu::begin_evict(uint64_t va) {
    const uint64_t vpn = va >> SG_PAGE_SHIFT;
    if (vpn >= npages_) return 0;

    std::lock_guard<std::mutex> g(map_lock_);
    const uint64_t cur = pte_[vpn].load(std::memory_order_acquire);
    if ((cur & (PTE_PRESENT | PTE_IN_VRAM)) != (PTE_PRESENT | PTE_IN_VRAM)) return 0;
    // exchange, not store: an engine's TLB hit may OR in DIRTY right now.
    const uint64_t old = pte_[vpn].exchange(PTE_PRESENT, std::memory_order_seq_cst);
    tlb_gen_.fetch_add(1, std::memory_order_seq_cst);
    tlb_invalidate_vpn(vpn);
    return old;
}

uint64_t Mmu::end_evict(uint64_t va) {
    const uint64_t vpn = va >> SG_PAGE_SHIFT;
    if (vpn >= npages_) return 0;
    std::lock_guard<std::mutex> g(map_lock_);
    return pte_[vpn].fetch_and(~(PTE_DIRTY | PTE_ACCESSED), std::memory_order_acq_rel) &
           (PTE_DIRTY | PTE_ACCESSED);
}

bool Mmu::test_and_clear_accessed(uint64_t va) {
    const uint64_t vpn = va >> SG_PAGE_SHIFT;
    if (vpn >= npages_) return false;
    return pte_[vpn].fetch_and(~PTE_ACCESSED, std::memory_order_acq_rel) & PTE_ACCESSED;
}

PageInfo Mmu::page(uint64_t va) const {
    PageInfo info;
    const uint64_t vpn = va >> SG_PAGE_SHIFT;
    if (vpn >= npages_) return info;
    const uint64_t pte = pte_[vpn].load(std::memory_order_acquire);
    if (!(pte & PTE_PRESENT)) return info;
    info.host = host_[vpn].load(std::memory_order_acquire);
    if (pte & PTE_IN_VRAM) {
        info.kind = PageInfo::Vram;
        info.pa = pte_pa(pte);
        return info;
    }
    info.kind = PageInfo::Host;
    return info;
}

int Mmu::unmap(uint64_t va, uint64_t npages) {
    if (va & (SG_PAGE_SIZE - 1)) return -EINVAL;
    if (npages == 0) return 0;
    const uint64_t vpn0 = va >> SG_PAGE_SHIFT;
    if (vpn0 >= npages_ || npages > npages_ - vpn0) return -EINVAL;

    std::lock_guard<std::mutex> g(map_lock_);
    for (uint64_t i = 0; i < npages; ++i) {
        pte_[vpn0 + i].store(0, std::memory_order_release);
        host_[vpn0 + i].store(0, std::memory_order_relaxed);
    }
    // Publish the generation before the shootdown. An insert that stored
    // its tag after the shootdown re-reads the generation and clears it.
    tlb_gen_.fetch_add(1, std::memory_order_seq_cst);
    for (uint64_t i = 0; i < npages; ++i) tlb_invalidate_vpn(vpn0 + i);
    return 0;
}

void Mmu::tlb_invalidate_vpn(uint64_t vpn) {
    for (uint32_t e = 0; e < SG_MAX_ENGINES; ++e)
        for (uint32_t i = 0; i < SG_TLB_ENTRIES; ++i) {
            auto& tag = tlb_[e].entry[i].vpn;
            if (tag.load(std::memory_order_seq_cst) == vpn)
                tag.store(~0ull, std::memory_order_release);
        }
}

void Mmu::reset_stats() {
    for (uint32_t e = 0; e < SG_MAX_ENGINES; ++e) {
        tlb_hits_[e].store(0, std::memory_order_relaxed);
        tlb_misses_[e].store(0, std::memory_order_relaxed);
    }
}

void Mmu::tlb_insert(uint32_t engine, uint64_t vpn, uint64_t pte) {
    EngineTlb& tlb = tlb_[engine];
    const uint64_t gen = tlb_gen_.load(std::memory_order_acquire);
    TlbEntry* slot = &tlb.entry[0];
    uint64_t oldest = ~0ull;
    for (uint32_t i = 0; i < SG_TLB_ENTRIES; ++i) {
        // relaxed: this engine is the only thread that fills its TLB, and
        // a concurrent shootdown only stores ~0, which we treat as empty.
        if (tlb.entry[i].vpn.load(std::memory_order_relaxed) == ~0ull) {
            slot = &tlb.entry[i];
            break;
        }
        if (tlb.entry[i].last_used < oldest) {
            oldest = tlb.entry[i].last_used;
            slot = &tlb.entry[i];
        }
    }
    // Tag is empty while the new PTE is stored, so a reader cannot pair
    // this PTE with the previous VPN. The generation re-checks catch a
    // mapping change that landed after we sampled `gen`; the tag store and
    // the last re-check are seq_cst to pair with the shootdown's tag loads.
    slot->vpn.store(~0ull, std::memory_order_relaxed);
    slot->pte.store(pte, std::memory_order_relaxed);
    slot->last_used = tlb.clock++;
    constexpr uint64_t kResident = PTE_PRESENT | PTE_IN_VRAM;
    if ((pte_[vpn].load(std::memory_order_acquire) & kResident) != kResident) return;
    if (tlb_gen_.load(std::memory_order_acquire) != gen) return;
    slot->vpn.store(vpn, std::memory_order_seq_cst);
    if (tlb_gen_.load(std::memory_order_seq_cst) != gen)
        slot->vpn.store(~0ull, std::memory_order_release);
}

Translate Mmu::walk(uint32_t engine, uint64_t va, bool write) {
    const uint64_t vpn = va >> SG_PAGE_SHIFT;
    if (vpn >= npages_) return {Translate::NotPresent, 0};

    uint64_t pte = pte_[vpn].load(std::memory_order_acquire);
    if (!(pte & PTE_PRESENT)) return {Translate::NotPresent, 0};
    if (!(pte & PTE_IN_VRAM)) return {Translate::HostResident, 0};

    const uint64_t bits = PTE_ACCESSED | (write ? PTE_DIRTY : 0);
    if ((pte & bits) != bits)
        pte = pte_[vpn].fetch_or(bits, std::memory_order_acq_rel) | bits;
    if (!(pte & PTE_PRESENT)) return {Translate::NotPresent, 0};
    if (!(pte & PTE_IN_VRAM)) return {Translate::HostResident, 0};

    tlb_insert(engine, vpn, pte);
    return {Translate::Ok, pte_pa(pte)};
}

Translate Mmu::translate(uint32_t engine, uint64_t va, bool write) {
    const uint64_t vpn = va >> SG_PAGE_SHIFT;
    EngineTlb& tlb = tlb_[engine];

    for (uint32_t i = 0; i < SG_TLB_ENTRIES; ++i) {
        TlbEntry& e = tlb.entry[i];
        if (e.vpn.load(std::memory_order_acquire) != vpn) continue;
        uint64_t pte = e.pte.load(std::memory_order_acquire);
        // The slot may have been reused between the two loads.
        if (e.vpn.load(std::memory_order_acquire) != vpn) break;
        if (!(pte & PTE_PRESENT) || !(pte & PTE_IN_VRAM)) break;
        if (write && !(pte & PTE_DIRTY)) {
            pte |= PTE_DIRTY;
            if (vpn < npages_) pte_[vpn].fetch_or(PTE_DIRTY, std::memory_order_relaxed);
            e.pte.store(pte, std::memory_order_relaxed);
        }
        e.last_used = tlb.clock++;
        tlb_hits_[engine].fetch_add(1, std::memory_order_relaxed);
        return {Translate::Ok, pte_pa(pte)};
    }

    tlb_misses_[engine].fetch_add(1, std::memory_order_relaxed);
    return walk(engine, va, write);
}

} // namespace softgpu::device
