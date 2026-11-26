#pragma once

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <vector>

namespace minifaiss {

inline constexpr std::size_t kCacheLineAlignment = 64;

template <typename T, std::size_t Alignment = kCacheLineAlignment>
struct AlignedAllocator {
    using value_type = T;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;

    template <typename U>
    explicit AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) {
            return nullptr;
        }
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        const std::size_t bytes = n * sizeof(T);
        const std::size_t padded_bytes =
            ((bytes + Alignment - 1) / Alignment) * Alignment;

#if defined(_MSC_VER)
        void* ptr = _aligned_malloc(padded_bytes, Alignment);
        if (!ptr) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
#else
        void* ptr = std::aligned_alloc(Alignment, padded_bytes);
        if (!ptr) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
#endif
    }

    void deallocate(T* ptr, std::size_t) noexcept {
        if (!ptr) {
            return;
        }
#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }

    template <typename U, std::size_t OtherAlignment>
    bool operator==(const AlignedAllocator<U, OtherAlignment>&) const noexcept {
        return Alignment == OtherAlignment;
    }
};

using AlignedFloatVector = std::vector<float, AlignedAllocator<float, kCacheLineAlignment>>;

}  // namespace minifaiss
