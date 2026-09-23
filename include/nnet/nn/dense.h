#ifndef DENSE_H
#define DENSE_H

#include "nnet/nn/parameter.h"
#include "nnet/core/tensor_ops.h"

#include <cstddef>
#include <vector>

namespace nnet {

	// One fully connected layer: every input connects to every output.
	//
	// It owns its weight and bias and nothing else. It does not apply an
	// activation, and it does not keep any value from a previous call
	struct Dense {
		Parameter weight;
		Parameter bias;

		// Both shapes come from the same two numbers, so the weight and the
		// bias cannot end up sized for different layers.
		Dense(std::size_t inputs, std::size_t outputs)
		: weight(createWeight(inputs, outputs)),
		  bias(createBias(outputs)) {}

		// Non-owning pointers to this layer's trainable values. The layer
		// keeps ownership; this just says where they live.
		std::vector<Parameter*> parameters() {
			return {&weight, &bias};
		}

		// Multiply by the weights, then add the bias. That is the whole
		// layer.

		Tensor forward(const Tensor& input) const {
			Tensor output = input * weight.value;
			addBias(output, bias.value);
			return output;
		}

		Tensor backward(const Tensor& layerInput, const Tensor& gradientFromAbove) {
			// One weight is shared by every group in the input, so when the
			// input has leading dimensions each group contributes its own gradient
			weight.grad = sumLeadingDimensions(
				matmulGradWeight(layerInput, gradientFromAbove),
				weight.value.rank());
			bias.grad = addBiasGrad(gradientFromAbove, bias.value);
			return matmulGradInput(gradientFromAbove, weight.value);
		}
	};
}

#endif
