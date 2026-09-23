#include "nnet/core/tensor_ops.h"
#include <algorithm>
#include <cmath>
#include <utility>

namespace nnet {

	// External Use Helpers

	Tensor fill(const Tensor::Shape& shape, double value) {
		std::size_t total = 1;
		for (const auto& dim : shape) {
			total *= dim;
		}
		return Tensor(shape, std::vector<double>(total, value));
	}


	Tensor zeros(const Tensor::Shape& shape) {
		return fill(shape, 0.0);
	}

	Tensor sigmoid(const Tensor& tensor) {
		const auto& input_data = tensor.getData();
		std::vector<double> out(input_data.size());
		for (size_t i = 0; i < input_data.size(); ++i) {
			out[i] = 1 / (1 + std::exp(-input_data[i]));
		}
		return Tensor(tensor.shape(), out);
	}

	// Internal Use Helpers

	Tensor sigmoidDerivitive(const Tensor& tensor) {
		const Tensor activated = sigmoid(tensor);
		const auto& activatedData = activated.getData();
		std::vector<double> grad(activatedData.size());
		for (std::size_t i = 0; i < activatedData.size(); i++) {
			grad[i] = activatedData[i] * (1 - activatedData[i]);
		}
		return Tensor(tensor.shape(), grad);
	}

	double sum(const Tensor& tensor) {
		const auto& input_data = tensor.getData();
		double total = 0.0;
		for (const auto& value : input_data) {
			total += value;
		}
		return total;
	}

	void addBias(Tensor& tensor, const Tensor& bias) {

		if (bias.rank() != 1)
			throw std::invalid_argument("Bias must be rank 1");

		const std::size_t width = bias.shape()[0];

		if (tensor.shape()[tensor.rank() - 1] != width) {
			throw std::invalid_argument("Bias length must equal the last dimension of the tensor");
		}

		// Whatever the rank, the tensor is a stack of rows `width` wide.
		// Every leading dimension is a batch dimension and is walked by the
		// odometer at the bottom of the loop.
		const std::size_t rows = tensor.numel() / width;
		Tensor::Shape indices(tensor.rank(), 0);

		for (std::size_t row = 0; row < rows; row++) {
			for (std::size_t col = 0; col < width; col++) {
				indices[tensor.rank() - 1] = col;
				tensor.at(indices) += bias.at({col});
			}

			// Advance every axis except the last, carrying like an odometer.
			for (std::size_t axis = tensor.rank() - 1; axis-- > 0;) {
				if (++indices[axis] < tensor.shape()[axis]) break;
				indices[axis] = 0;
			}
		}
	}

	// Gradient of addBias. The same bias element is added to every row of the
	// matrix, so bias element c influences every output in column c. The chain
	// rule therefore sums the upstream gradient down that whole column.
	// Shapes: upstreamGrad is [rows, columns], bias is [columns], result is [columns].
	Tensor addBiasGrad(const Tensor& upstreamGrad, const Tensor& bias) {
		if (bias.rank() != 1)
			throw std::invalid_argument("addBiasGrad requires a rank-1 bias");

		const std::size_t width = bias.shape()[0];

		if (upstreamGrad.shape()[upstreamGrad.rank() - 1] != width)
			throw std::invalid_argument("Bias length must equal the last dimension of the upstream gradient");

		// The same bias value is reused once per row, at every leading
		// dimension, so its gradient is the sum down that whole column.
		const auto& data = upstreamGrad.getData();
		std::vector<double> grad(width, 0.0);
		for (std::size_t i = 0; i < data.size(); i++) {
			grad[i % width] += data[i];
		}
		return Tensor(bias.shape(), std::move(grad));
	}

	// Swaps the last two axes. Every leading dimension is a batch dimension
	// and is left exactly as it is, matching what matmul already does.
	// Copies rather than returning a view, since a Tensor owns its storage.
	Tensor transpose(const Tensor& tensor) {
		if (tensor.rank() < 2)
			throw std::invalid_argument("transpose requires rank 2 or higher");

		const std::size_t rows = tensor.shape()[tensor.rank() - 2];
		const std::size_t columns = tensor.shape()[tensor.rank() - 1];
		const std::size_t matrices = tensor.numel() / rows / columns;

		const auto& data = tensor.getData();
		std::vector<double> values(tensor.numel());

		for (std::size_t matrix = 0; matrix < matrices; matrix++) {
			const std::size_t start = matrix * rows * columns;
			for (std::size_t row = 0; row < rows; row++) {
				for (std::size_t col = 0; col < columns; col++) {
					values[start + col * rows + row] =
						data[start + row * columns + col];
				}
			}
		}

		Tensor::Shape outputShape = tensor.shape();
		std::swap(outputShape[tensor.rank() - 2], outputShape[tensor.rank() - 1]);
		return Tensor(std::move(outputShape), std::move(values));
	}

	// Adds up the leading dimensions until only `targetRank` axes remain.
	//
	// This is what a shared value needs. When one weight is used by every
	// group in a batch, each group produces its own contribution to that
	// weight's gradient, and the contributions have to be added together.
	// addBiasGrad does the same thing for the same reason.
	Tensor sumLeadingDimensions(const Tensor& tensor, std::size_t targetRank) {
		if (targetRank == 0 || targetRank > tensor.rank())
			throw std::invalid_argument(
				"targetRank must be between 1 and the tensor's own rank");

		if (targetRank == tensor.rank())
			return tensor;

		Tensor::Shape outputShape(tensor.shape().end() - static_cast<long>(targetRank),
		                          tensor.shape().end());

		std::size_t outputSize = 1;
		for (const auto& dimension : outputShape) {
			outputSize *= dimension;
		}

		const auto& data = tensor.getData();
		std::vector<double> values(outputSize, 0.0);
		for (std::size_t i = 0; i < data.size(); i++) {
			values[i % outputSize] += data[i];
		}
		return Tensor(std::move(outputShape), std::move(values));
	}

	// For Z = X * W, the weight gradient is X-transpose * dL/dZ.
	// Shapes: X is [M, K] and dL/dZ is [M, N], so X-transpose is [K, M] and the
	// product is [K, N], which is exactly W's shape. Rank and inner-dimension
	Tensor matmulGradWeight(const Tensor& input, const Tensor& upstreamGrad) {
		return transpose(input) * upstreamGrad;
	}

	// For Z = X * W, the input gradient is dL/dZ * W-transpose.
	// Shapes: dL/dZ is [M, N] and W is [K, N], so W-transpose is [N, K] and the
	// product is [M, K], which is exactly X's shape.
	Tensor matmulGradInput(const Tensor& upstreamGrad, const Tensor& weight) {
		return upstreamGrad * transpose(weight);
	}

	double binaryCrossEntropy(const Tensor& predictions, const Tensor& targets) {
		if (predictions.shape() != targets.shape()) {
			throw std::invalid_argument("Predictions and targets must have the same shape");
		}

		const auto& predictionData = predictions.getData();
		const auto& targetData = targets.getData();

		double loss = 0.0;
		for (std::size_t i = 0; i < predictionData.size(); i++) {
			const double probability = std::clamp(predictionData[i], 1.0e-15, 1.0 - 1.0e-15);
			loss += -(targetData[i] * std::log(probability) + (1 - targetData[i]) * std::log(1 - probability));
		}
		return loss / predictionData.size();
	}

	Tensor binaryCrossEntropyGrad(const Tensor& predictions, const Tensor& targets) {
		if (predictions.shape() != targets.shape()) {
			throw std::invalid_argument("Predictions and targets must have the same shape");
		}

		const auto& predictionData = predictions.getData();
		const auto& targetData = targets.getData();

		std::vector<double> grad(predictionData.size());
		for (std::size_t i = 0; i < predictionData.size(); i++) {
			const double probability = std::clamp(predictionData[i], 1.0e-15, 1.0 - 1.0e-15);
			grad[i] = ((probability - targetData[i]) / (probability * (1 - probability))) / predictionData.size();
		}
		return Tensor(predictions.shape(), grad);
	}


	Tensor createBias(std::size_t layerCount) {
		return zeros(Tensor::Shape({layerCount}));
	}

	Tensor createWeight(const size_t numInputs, const size_t numOutputs) {
		std::vector<double> weightData(numInputs * numOutputs);
		for (size_t i = 0; i < weightData.size(); ++i) {
			weightData[i] = ((double)rand() / RAND_MAX) - 0.5; // Random values between -0.5 and 0.5
		}
		return Tensor({numInputs, numOutputs}, weightData);
	}

	std::vector<Tensor> createWeights(std::vector<size_t> layerSizes) {
		std::vector<Tensor> vectorOfWeightTensors;
		vectorOfWeightTensors.reserve(layerSizes.size() - 1);
		for (size_t i = 0; i + 1 < layerSizes.size(); i++) {
			vectorOfWeightTensors.push_back(createWeight(layerSizes[i], layerSizes[i + 1]));
		}
		return vectorOfWeightTensors;
	}

	

}


