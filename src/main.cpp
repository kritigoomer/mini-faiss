#include "knn_index.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

std::vector<float> make_random_vectors(std::size_t count,
                                       std::size_t dim,
                                       std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> out(count * dim);
    for (float& x : out) {
        x = dist(rng);
    }
    return out;
}

bool verify_results(const std::vector<minifaiss::Neighbor>& simd_res,
                    const std::vector<minifaiss::Neighbor>& scalar_res,
                    float tol = 1e-3f) {
    if (simd_res.size() != scalar_res.size()) {
        return false;
    }
    for (std::size_t i = 0; i < simd_res.size(); ++i) {
        if (simd_res[i].id != scalar_res[i].id) {
            return false;
        }
        if (std::fabs(simd_res[i].distance_sq - scalar_res[i].distance_sq) > tol) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    constexpr std::size_t kNumVectors = 100'000;
    constexpr std::size_t kDim = 768;
    constexpr std::size_t kTopK = 10;
    constexpr std::size_t kNumQueries = 50;

    std::cout << "Generating " << kNumVectors << " vectors of dim " << kDim
              << " (" << std::fixed << std::setprecision(1)
              << (static_cast<double>(kNumVectors * kDim * sizeof(float)) / (1024.0 * 1024.0))
              << " MiB)...\n";

    auto dataset = make_random_vectors(kNumVectors, kDim, 42ULL);
    auto queries = make_random_vectors(kNumQueries, kDim, 1337ULL);

    minifaiss::ExactKnnIndex index(kDim);

    // Ingest in two 50k batches to exercise batch ingestion
    const std::size_t half = kNumVectors / 2;
    index.add_vectors(dataset.data(), half);
    index.add_vectors(dataset.data() + half * kDim, kNumVectors - half);

    // 1. Correctness check on first query
    const float* q0 = queries.data();
    auto simd_top = index.search(q0, kTopK);
    auto scalar_top = index.search_scalar_baseline(q0, kTopK);

    if (!verify_results(simd_top, scalar_top)) {
        std::cerr << "Mismatch between SIMD and scalar top-" << kTopK << " results!\n";
        return 1;
    }
    std::cout << "Correctness check passed (top-" << kTopK
              << " IDs and L2^2 distances match scalar baseline).\n\n";

    // 2. Benchmark scalar baseline
    double scalar_total_ms = 0.0;
    std::uint64_t sink = 0;
    for (std::size_t q = 0; q < kNumQueries; ++q) {
        const float* qp = queries.data() + q * kDim;
        auto t0 = std::chrono::steady_clock::now();
        auto res = index.search_scalar_baseline(qp, kTopK);
        auto t1 = std::chrono::steady_clock::now();
        scalar_total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        sink += res.front().id;
    }

    // 3. Benchmark SIMD + contiguous heap search
    double simd_total_ms = 0.0;
    for (std::size_t q = 0; q < kNumQueries; ++q) {
        const float* qp = queries.data() + q * kDim;
        auto t0 = std::chrono::steady_clock::now();
        auto res = index.search(qp, kTopK);
        auto t1 = std::chrono::steady_clock::now();
        simd_total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        sink += res.front().id;
    }

    const double scalar_avg_ms = scalar_total_ms / kNumQueries;
    const double simd_avg_ms = simd_total_ms / kNumQueries;
    const double speedup = scalar_avg_ms / simd_avg_ms;

    std::cout << std::setprecision(3);
    std::cout << "Scalar baseline avg latency : " << scalar_avg_ms << " ms / query\n";
    std::cout << "SIMD + heap avg latency     : " << simd_avg_ms << " ms / query\n";
    std::cout << "Speedup                     : " << speedup << "x\n";
    std::cout << "(Checksum: " << sink << ")\n";

    return 0;
}


//=================================================================================

//#include "index.hpp"
//#include <iostream>
//#include <vector>

//int main() {
//	size_t dim = 3;
//	FlatIndex index(dim);
//
//	index.add({1.0f, 2.0f, 3.0f});
//	index.add({1.1f, 2.1f, 3.1f});
//	index.add({10.0f, 20.0f, 30.0f});
//
//	std::cout << "Successfully added " 
//		<< index.size() << 
//		" vectors to the index.\n";
//
//	std::vector<float> query = {1.05f, 2.05f, 3.05f};
//	size_t k = 2;
//
//	auto results = index.search(query, k);
//
//	std::cout << "\nSearch Results for query [1.05, 2.05, 3.05]:\n";
//	for (const auto& [id, dist] : results) {
//		std::cout << " -> Vector ID: " << id << "\t| Distance: " 
//			<< dist << "\n";
//	}
//
//	return 0;
//}
