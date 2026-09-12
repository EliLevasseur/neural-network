#include "nnet/core/tensor_ops.h"
#include <cmath>

namespace nnet {

	Tensor sigmoid(const Tensor& tensor) {
		const auto& input_data = tensor.getData();
		std::vector<double> out(input_data.size());
		for (size_t i = 0; i < input_data.size(); ++i) {
			out[i] = 1 / (1 + std::exp(-input_data[i]));
		}
		return Tensor(tensor.shape(), out);	
	}

	double sum(const Tensor& tensor) {
		const auto& input_data = tensor.getData();
		double total = 0.0;
		for (const auto& value : input_data) {
			total += value;
		}
		return total;
	}

	int calculateNodes(const Tensor::Shape& layers) {
		int totalNodes = 0;
		for (const auto& layer : layers) {
			totalNodes += layer;
		}
		return totalNodes;
	}

	std::vector<Tensor> createWeights(const Tensor& tensor, const Tensor::Shape& networkShape) {
		std::vector<Tensor> weights;
		std::size_t inputSize = tensor.numel();

		for (std::size_t outputSize : networkShape) {
			std::vector<double> values(outputSize * inputSize);
			for (double& value : values) {
				value = (static_cast<double>(rand()) / RAND_MAX) - 0.5;
			}

			weights.emplace_back(
				Tensor::Shape{outputSize, inputSize},
				std::move(values)
			);
			inputSize = outputSize;
		}

		return weights;
	}

	Tensor createBiases(const Tensor& tensor) {
		std::vector<double> data(tensor.numel(), 0.0);
		return Tensor(tensor.shape(), data);
	}

	Tensor forwardPass(const Tensor& input, const Tensor& weights, const Tensor& biases, const Tensor::Shape& layers) {
		

	}



}