#pragma once

#include "aligned_buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace minifaiss {

struct Neighbor {
    float distance_sq = 0.0f;
    std::uint32_t id = 0;

    // Max-heap ordering: largest distance sits at top() so we evict the farthest
    // candidate when a closer neighbor is found.
    bool operator<(const Neighbor& other) const noexcept {
        if (distance_sq != other.distance_sq) {
            return distance_sq < other.distance_sq;
        }
        return id < other.id;
    }
};

class ExactKnnIndex {
public:
    explicit ExactKnnIndex(std::size_t dim);

    // Appends a batch of contiguous row-major vectors: size must be a multiple of dim.
    void add_vectors(const float* data, std::size_t num_vectors);
    void add_vectors(const std::vector<float>& data);

    // Exact k-NN search using SIMD L2 kernel + bounded max-heap of size k.
    [[nodiscard]] std::vector<Neighbor> search(const float* query, std::size_t k) const;

    // Scalar baseline search (no SIMD intrinsics) for benchmarking and verification.
    [[nodiscard]] std::vector<Neighbor> search_scalar_baseline(const float* query,
                                                               std::size_t k) const;

    [[nodiscard]] std::size_t dimension() const noexcept { return dim_; }
    [[nodiscard]] std::size_t size() const noexcept { return num_vectors_; }
    [[nodiscard]] std::size_t memory_bytes() const noexcept {
        return data_.size() * sizeof(float);
    }

    // Exposed for direct kernel unit testing and benchmarking.
    [[nodiscard]] static float compute_l2_sq_scalar(const float* a,
                                                    const float* b,
                                                    std::size_t dim) noexcept;
    [[nodiscard]] static float compute_l2_sq_simd(const float* a,
                                                  const float* b,
                                                  std::size_t dim) noexcept;

private:
    std::size_t dim_ = 0;
    std::size_t num_vectors_ = 0;
    AlignedFloatVector data_;
};

}  // namespace minifaiss
