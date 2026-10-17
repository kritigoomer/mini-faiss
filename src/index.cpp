#include "index.hpp" // "" mean to look in project directory
#include <cmath> // gives std::sqrt()
#include <algorithm> // gives std::sort()
#include <stdexcept> // gives error handling -> std::invalid_argument

// Constructor
// dimension_(dimension) -- Member Initializer List 
// 	faster, best practice; initializes directly to passed value
// {} means no extra code
FlatIndex::FlatIndex(size_t dimension) : dimension_(dimension) {}

// No return type
void FlatIndex::add(const std::vector<float>& vector) {
	if (vector.size() != dimension_) { // 4D vector into 3D db will crash
					   // math later, throws an error here
		throw std::invalid_argument
			("Vector dimension mistmatch!");
	}
	vectors_.push_back(vector); // copies incoming vector and appends
				    // to end of 2D vectors_ storage
}

// Compute L2 Euclidean Distance
float FlatIndex::compute_l2_distance(
		const std::vector<float>& a,
		const std::vector<float>& b) const {
	float sum = 0.0f; // f shows single precision floating-point number
	for (size_t i = 0; i < dimension_; ++i) {
		float diff = a[i] - b[i];
		sum += diff * diff;
	}
	return std::sqrt(sum);
}

// Search Function
std::vector<std::pair<size_t, float>> FlatIndex::search(
		const std::vector<float>& query, size_t k) const {
	if (query.size() != dimension_) { // dimension check
		throw std::invalid_argument("Query dimension mismatch!");
	}

	std::vector<std::pair<size_t, float>> distances; // temporary list

	distances.reserve(vectors_.size()); // optimization
					    // allocate memory for all vectors
					    // right now, prevents constantly
					    // resizing array on the fly
					    // saves CPU cycles

	for (size_t i = 0; i < vectors_.size(); ++i) { // loops through
						       // every vector stored
		float dist = compute_l2_distance(query, vectors_[i]); // distance
		distances.emplace_back(i, dist); // bundles vector's id (i) and 
						 // calculated distance tg
						 // and drops them into distances
						 // temporary list
	}

	std::sort(
		distances.begin(), 
		distances.end(), 
		[](const auto& a, const auto& b) { // lambda function
						   // anonymous inline function
						   // compare item a and item b
						   // based on their second value
						   // (distance) and sort them in
						   // ascending order; smallest
						   // distance = closest match
			return a.second < b.second;
	});
	
	if (k < distances.size()) { // trims list down to top k matches 
				    // if k < size
		distances.resize(k);
	}

	return distances; // sends final results back
}

size_t FlatIndex::size() const {
	return vectors_.size();
}
