#include "index.hpp"
#include <iostream>
#include <vector>

int main() {
	size_t dim = 3;
	FlatIndex index(dim);

	index.add({1.0f, 2.0f, 3.0f});
	index.add({1.1f, 2.1f, 3.1f});
	index.add({10.0f, 20.0f, 30.0f});

	std::cout << "Successfully added " 
		<< index.size() << 
		" vectors to the index.\n";

	std::vector<float> query = {1.05f, 2.05f, 3.05f};
	size_t k = 2;

	auto results = index.search(query, k);

	std::cout << "\nSearch Results for query [1.05, 2.05, 3.05]:\n";
	for (const auto& [id, dist] : results) {
		std::cout << " -> Vector ID: " << id << "\t| Distance: " 
			<< dist << "\n";
	}

	return 0;
}
