#ifndef DENSE_H
#define DENSE_H

#include "module.h"

#include <cstddef>
#include <vector>

namespace nnet {

	// One fully connected layer: every input connects to every output.
	// It owns its weight and bias and nothing else. It does not apply an
	// activation, and it does not keep any value from a previous call
	class Dense : public Unary_Module {
		public:
			Parameter weight;
			Parameter bias;

			// Both shapes come from the same two numbers, so the weight and the
			// bias cannot end up sized for different layers.
			Dense(std::size_t inputs, std::size_t outputs)
			: weight(createWeight(inputs, outputs)),
			bias(createBias(outputs)) {}

    

		protected:
            Value forward_impl(const Value& input) const override {
                // Each call gets its own leaves. A Dense used twice in one pass
                // makes two pairs, both pointing at the same Parameter, so both
                // contributions land in its gradient.
                Value product = matmul(input, leafFor(weight));
                return addBias(product, leafFor(bias));
            }

   			std::vector<NamedParameter> localNamedParameters() override {
			return {
				{"weight", &weight},
				{"bias", &bias}
			};
		}

		
	};
}

#endif
