#include "driver/vram_alloc.h"

namespace softgpu::driver {

VramAllocator::VramAllocator(uint64_t base, uint64_t size, uint64_t align) : align_(align) {
    free_.emplace(base, size);
}

std::optional<uint64_t> VramAllocator::alloc(uint64_t bytes) {
    if (bytes == 0) return std::nullopt;
    const uint64_t need = (bytes + align_ - 1) / align_ * align_;
    for (auto it = free_.begin(); it != free_.end(); ++it) {
        if (it->second < need) continue;
        const uint64_t addr = it->first;
        const uint64_t remain = it->second - need;
        free_.erase(it);
        if (remain) free_.emplace(addr + need, remain);
        live_.emplace(addr, need);
        live_bytes_ += need;
        return addr;
    }
    return std::nullopt;
}

bool VramAllocator::free(uint64_t addr) {
    uint64_t size = 0;
    if (!detach(addr, &size)) return false;
    release(addr, size);
    return true;
}

bool VramAllocator::detach(uint64_t addr, uint64_t* size_out) {
    auto it = live_.find(addr);
    if (it == live_.end()) return false;
    *size_out = it->second;
    live_.erase(it);
    live_bytes_ -= *size_out;
    return true;
}

void VramAllocator::release(uint64_t addr, uint64_t len) {
    uint64_t start = addr;
    // Coalesce with the following block, then with the preceding one.
    auto next = free_.lower_bound(start);
    if (next != free_.end() && next->first == start + len) {
        len += next->second;
        next = free_.erase(next);
    }
    if (next != free_.begin()) {
        auto prev = std::prev(next);
        if (prev->first + prev->second == start) {
            start = prev->first;
            len += prev->second;
            free_.erase(prev);
        }
    }
    free_.emplace(start, len);
}

bool VramAllocator::owns_range(uint64_t addr, uint64_t len) const {
    // Find the live allocation starting at or before addr. Linear over live
    // allocations is fine for the baseline; this is a correctness check, not
    // a hot path yet — and when it becomes one, that is a measurable decision.
    for (const auto& [start, size] : live_) {
        if (addr >= start && addr - start <= size && len <= size - (addr - start)) return true;
    }
    return false;
}

} // namespace softgpu::driver
