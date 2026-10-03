#pragma once
// First-fit free-list allocator over [base, base + size).
//
// BASELINE: simplest correct thing. No size classes, no per-context pools,
// O(#free blocks) allocation. The driver uses one instance for VRAM frames
// and one for the managed VA range, each under its own lock.

#include <cstdint>
#include <map>
#include <optional>
#include <unordered_map>

namespace softgpu::driver {

class VramAllocator {
public:
    VramAllocator(uint64_t base, uint64_t size, uint64_t align);

    std::optional<uint64_t> alloc(uint64_t bytes);
    bool free(uint64_t addr); // detach + release

    // Two-phase free for memory that in-flight work may still reference:
    // detach() removes it from the live set (owns_range() stops matching)
    // without recycling it; release() returns it to the free list later.
    bool detach(uint64_t addr, uint64_t* size_out);
    void release(uint64_t addr, uint64_t size);

    // True if [addr, addr+len) lies entirely inside one live allocation.
    bool owns_range(uint64_t addr, uint64_t len) const;

    uint64_t bytes_live() const { return live_bytes_; }

private:
    uint64_t align_;
    std::map<uint64_t, uint64_t> free_;           // start -> length, coalesced
    std::unordered_map<uint64_t, uint64_t> live_; // start -> length
    uint64_t live_bytes_ = 0;
};

} // namespace softgpu::driver
