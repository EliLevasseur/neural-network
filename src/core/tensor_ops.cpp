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

		if (bias.shape()[bias.rank() - 1] != tensor.shape()[tensor.rank() - 1]) {
			throw std::invalid_argument("Bias must be tensor one dimension less than the input tensor with the same size as the last dimension of the input tensor");
		}
		
		for (size_t row = 0; row < tensor.shape()[tensor.rank() - 2]; row++) {
			for (size_t col = 0; col < tensor.shape()[tensor.rank() - 1]; col++) {
				tensor.at({row, col}) += bias.at({col});
			}
		}
	}

	// Gradient of addBias. The same bias element is added to every row of the
	// matrix, so bias element c influences every output in column c. The chain
	// rule therefore sums the upstream gradient down that whole column.
	// Shapes: upstreamGrad is [rows, columns], bias is [columns], result is [columns].
	Tensor addBiasGrad(const Tensor& upstreamGrad, const Tensor& bias) {
		if (upstreamGrad.rank() != 2)
			throw std::invalid_argument("addBiasGrad requires a rank-2 upstream gradient");

		if (bias.rank() != 1)
			throw std::invalid_argument("addBiasGrad requires a rank-1 bias");

		const std::size_t rows = upstreamGrad.shape()[0];
		const std::size_t columns = upstreamGrad.shape()[1];

		if (bias.shape()[0] != columns)
			throw std::invalid_argument("Bias length must equal the last dimension of the upstream gradient");

		std::vector<double> grad(columns, 0.0);
		for (std::size_t row = 0; row < rows; row++) {
			for (std::size_t col = 0; col < columns; col++) {
				grad[col] += upstreamGrad.at({row, col});
			}
		}
		return Tensor(bias.shape(), std::move(grad));
	}

	// Rank-2 only. This physically copies the elements instead of returning a
	// Shapes: input is [rows, columns], result is [columns, rows].
	Tensor transpose(const Tensor& tensor) {
		if (tensor.rank() != 2)
			throw std::invalid_argument("transpose requires a rank-2 tensor");

		const std::size_t rows = tensor.shape()[0];
		const std::size_t columns = tensor.shape()[1];

		std::vector<double> values(tensor.numel());
		for (std::size_t row = 0; row < rows; row++) {
			for (std::size_t col = 0; col < columns; col++) {
				// Input element [row, col] becomes output element [col, row]. The
				// output has `rows` columns, so that lands at flat position
				// col * rows + row.
				values[col * rows + row] = tensor.at({row, col});
			}
		}
		return Tensor(Tensor::Shape{columns, rows}, std::move(values));
	}

	// For Z = X * W, the weight gradient is X-transpose * dL/dZ.
	// Shapes: X is [M, K] and dL/dZ is [M, N], so X-transpose is [K, M] and the
	// product is [K, N], which is exactly W's shape. Rank and inner-dimension
	// errors are raised by transpose and by operator* respectively.
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
		return zeros(Tensor::Shape({1, layerCount}));
	}

	Tensor createWeight(const size_t numInputs, const size_t numOutputs) {
		std::vector<double> weightData(numInputs * numOutputs);
		for (size_t i = 0; i < weightData.size(); ++i) {
			weightData[i] = ((double)rand() / RAND_MAX) - 0.5; // Random values between -0.5 and 0.5
		}
		return Tensor({numInputs, numOutputs}, weightData);
	}

	std::vector<Paramater> createWeights(std::vector<size_t> layerSizes) {
		std::vector<Paramater> vectorOfWeightTensors;
		vectorOfWeightTensors.reserve(layerSizes.size() - 1);
		for (size_t i = 0; i + 1 < layerSizes.size(); i++) {
			vectorOfWeightTensors.emplace_back(createWeight(layerSizes[i], layerSizes[i + 1]));
		}
		return vectorOfWeightTensors;
	}

	Tensor feedForward(Tensor inputTensor, std::vector<Paramater>& weightParamaters) {
		Tensor output = sigmoid(inputTensor * weightParamaters[0].value);
		for (size_t tensor = 1; tensor < weightParamaters.size(); tensor++) {
			output = sigmoid(output * weightParamaters[tensor].value);
		}
		return output;

	}
}


