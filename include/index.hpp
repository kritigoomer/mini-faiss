// include guard -- include file once; implementations can only be done once
#pragma once 
#include <vector> // dynamic array template std::vector
#include <utility> // std::pair, lets bundle two values together

class FlatIndex {
private:
	// size_t is unsigned int used spec. for size and count in C++
	// stores dimension of vector (2, 3, 768)
	// private variable = _ (trailing underscore) 
	size_t dimension_;
	
	// 2D array (vector of vectors)
	// holds all embeddings added to database
	// will flatten this 
	std::vector<std::vector<float>> vectors_;
	
	
	// Private helper method to compute Euclidian (L2) distance
	// const shows this method will not modify any member variables
	// best practice in C++ for read-only functions
	float compute_l2_distance(
			const std::vector<float>& a, 
			const std::vector<float>& b
			) const;

public:
	// Constructor - runs automatically when you create your object
	// forces you to tell dimensions of your vectors
	// explicit: modern C++ best practice
	// 	prevents compiler from doing weird accidental implicit type
	// 	conversions behind your back.
	// 	FlatIndex index = 10; X :(
	// 	FlatIndex index(10); :)
	explicit FlatIndex(size_t dimension);

	// Add a vector to the database
	// passes vector by reference & (not a new physical copy -- slow!)
	// it is const because it will not be modified 
	// const - read only - best practice
	void add(const std::vector<float>& vector);

	// Search for the top-k nearest neighbors
	// Return Type: List of pairs
	// 	Pair:
	// 	- size_t : ID/index of stored vector
	// 	- float : its calculated distance to your query
	// Parameters:
	// 	- query vector
	// 	- k : how many top matches you want
	// const bc search doesn't change database contents
	std::vector<std::pair<size_t, float>> search(
			const std::vector<float>& query,
			size_t k
			) const;
	
	// Get no. of stored vectors
	// doesn't modify contents so const
	size_t size() const;
};
