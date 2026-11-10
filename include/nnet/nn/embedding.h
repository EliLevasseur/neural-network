#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "module.h"

#include <cstddef>
#include <random>
#include <vector>

namespace nnet {

	// Swaps each id for its own learned row of numbers.
	class Embedding : public Unary_Module {
		public:
			Parameter table;

			// One row per possible id. The generator is borrowed, like Dense's.
			Embedding(std::size_t idCount, std::size_t width, std::mt19937& generator)
			: table(createWeight(idCount, width, generator)) {}

		protected:
			Value forward_impl(const Value& ids) const override {
				return embedding(ids, leafFor(table));
			}

			std::vector<NamedParameter> localNamedParameters() override {
				return {
					{"table", &table}
				};
			}
	};
}

#endif
