#include "knn_index.hpp"

#include <algorithm>
#include <queue>
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace minifaiss {

namespace {

#if defined(__AVX2__)
float horizontal_sum_ps(__m256 v) noexcept {
    // v = [a0, a1, a2, a3, a4, a5, a6, a7]
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(low, high);          // [a0+a4, a1+a5, a2+a6, a3+a7]
    __m128 shuf = _mm_movehdup_ps(sum128);          // [a1+a5, a1+a5, a3+a7, a3+a7]
    __m128 sums = _mm_add_ps(sum128, shuf);         // [s0+s1, -, s2+s3, -]
    shuf = _mm_movehl_ps(shuf, sums);               // [s2+s3, -, -, -]
    sums = _mm_add_ss(sums, shuf);                  // [s0+s1+s2+s3, -, -, -]
    return _mm_cvtss_f32(sums);
}
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
float horizontal_sum_neon(float32x4_t v) noexcept {
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    float32x2_t pair_sum = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    float32x2_t total = vpadd_f32(pair_sum, pair_sum);
    return vget_lane_f32(total, 0);
#endif
}
#endif

// Updates a bounded max-heap of capacity k with a new (distance_sq, id) candidate.
inline void push_bounded_topk(std::priority_queue<Neighbor>& heap,
                              std::size_t k,
                              float dist_sq,
                              std::uint32_t id) {
    if (k == 0) {
        return;
    }
    if (heap.size() < k) {
        heap.push(Neighbor{dist_sq, id});
    } else if (dist_sq < heap.top().distance_sq) {
        heap.pop();
        heap.push(Neighbor{dist_sq, id});
    }
}

std::vector<Neighbor> drain_sorted(std::priority_queue<Neighbor>& heap) {
    std::vector<Neighbor> out(heap.size());
    for (std::size_t idx = out.size(); idx > 0; --idx) {
        out[idx - 1] = heap.top();
        heap.pop();
    }
    return out;  // Sorted ascending by distance_sq (closest first)
}

}  // namespace

ExactKnnIndex::ExactKnnIndex(std::size_t dim) : dim_(dim) {
    if (dim_ == 0) {
        throw std::invalid_argument("Embedding dimension must be greater than zero");
    }
}

void ExactKnnIndex::add_vectors(const float* raw_data, std::size_t count) {
    if (count == 0) {
        return;
    }
    if (!raw_data) {
        throw std::invalid_argument("Input vector pointer cannot be null");
    }

    const std::size_t added_floats = count * dim_;
    const std::size_t old_size = data_.size();
    data_.resize(old_size + added_floats);
    std::copy_n(raw_data, added_floats, data_.data() + old_size);
    num_vectors_ += count;
}

void ExactKnnIndex::add_vectors(const std::vector<float>& flat_batch) {
    if (flat_batch.size() % dim_ != 0) {
        throw std::invalid_argument("Batch size must be an exact multiple of dimension");
    }
    add_vectors(flat_batch.data(), flat_batch.size() / dim_);
}

float ExactKnnIndex::compute_l2_sq_scalar(const float* a,
                                          const float* b,
                                          std::size_t dim) noexcept {
    float acc = 0.0f;
    for (std::size_t j = 0; j < dim; ++j) {
        const float diff = a[j] - b[j];
        acc += diff * diff;
    }
    return acc;
}

float ExactKnnIndex::compute_l2_sq_simd(const float* a,
                                        const float* b,
                                        std::size_t dim) noexcept {
    std::size_t j = 0;

#if defined(__AVX2__)
    __m256 acc256 = _mm256_setzero_ps();
    for (; j + 8 <= dim; j += 8) {
        __m256 va = _mm256_loadu_ps(a + j);
        __m256 vb = _mm256_loadu_ps(b + j);
        __m256 diff = _mm256_sub_ps(va, vb);
#if defined(__FMA__)
        acc256 = _mm256_fmadd_ps(diff, diff, acc256);
#else
        acc256 = _mm256_add_ps(acc256, _mm256_mul_ps(diff, diff));
#endif
    }
    float total = horizontal_sum_ps(acc256);
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    float32x4_t acc128 = vdupq_n_f32(0.0f);
    for (; j + 4 <= dim; j += 4) {
        float32x4_t va = vld1q_f32(a + j);
        float32x4_t vb = vld1q_f32(b + j);
        float32x4_t diff = vsubq_f32(va, vb);
#if defined(__aarch64__) || defined(__ARM_FEATURE_FMA)
        acc128 = vfmaq_f32(acc128, diff, diff);
#else
        acc128 = vmlaq_f32(acc128, diff, diff);
#endif
    }
    float total = horizontal_sum_neon(acc128);
#else
    float total = 0.0f;
#endif

    // Tail loop for dimensions not divisible by SIMD lane width
    for (; j < dim; ++j) {
        const float diff = a[j] - b[j];
        total += diff * diff;
    }
    return total;
}

std::vector<Neighbor> ExactKnnIndex::search(const float* query, std::size_t k) const {
    if (!query) {
        throw std::invalid_argument("Query pointer cannot be null");
    }
    if (k == 0 || num_vectors_ == 0) {
        return {};
    }

    const std::size_t effective_k = std::min(k, num_vectors_);
    std::priority_queue<Neighbor> max_heap;

    const float* base = data_.data();
    for (std::size_t i = 0; i < num_vectors_; ++i) {
        const float* vec = base + i * dim_;
        const float dist_sq = compute_l2_sq_simd(query, vec, dim_);
        push_bounded_topk(max_heap, effective_k, dist_sq, static_cast<std::uint32_t>(i));
    }

    return drain_sorted(max_heap);
}

std::vector<Neighbor> ExactKnnIndex::search_scalar_baseline(const float* query,
                                                            std::size_t k) const {
    if (!query) {
        throw std::invalid_argument("Query pointer cannot be null");
    }
    if (k == 0 || num_vectors_ == 0) {
        return {};
    }

    const std::size_t effective_k = std::min(k, num_vectors_);
    std::priority_queue<Neighbor> max_heap;

    const float* base = data_.data();
    for (std::size_t i = 0; i < num_vectors_; ++i) {
        const float* vec = base + i * dim_;
        const float dist_sq = compute_l2_sq_scalar(query, vec, dim_);
        push_bounded_topk(max_heap, effective_k, dist_sq, static_cast<std::uint32_t>(i));
    }

    return drain_sorted(max_heap);
}

}  // namespace minifaiss
