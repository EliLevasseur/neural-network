#include "nnet/core/tensor_ops.h"
#include <algorithm>
#include <cmath>
#include <numeric>
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

	// ACTIVATIONS

	Tensor sigmoid(const Tensor& tensor) {
		const auto& input_data = tensor.getData();
		std::vector<double> out(input_data.size());
		for (size_t i = 0; i < input_data.size(); ++i) {
			out[i] = 1 / (1 + std::exp(-input_data[i]));
		}
		return Tensor(tensor.shape(), out);
	}

	Tensor relu(const Tensor& tensor) {
		const auto& input_data = tensor.getData();
		std::vector<double> out(input_data.size());
		for (size_t i = 0; i < input_data.size(); ++i) {
			out[i] = std::max(0.0, input_data[i]);
		}
		return Tensor(tensor.shape(), out);
	}

	Tensor softmax(const Tensor& scores) {
		const std::vector<double>& scoreData = scores.getData();
		// Each row is one image's scores. Every row gets its own largest score
		// and its own total, so images in the same batch never mix.
		const std::size_t width = scores.shape().back();
		const std::size_t rows = scores.numel() / width;
		std::vector<double> probabilities(scoreData.size());

		for (std::size_t row = 0; row < rows; row++) {
			const std::size_t rowStart = row * width;
			const auto rowBegin = scoreData.begin() + rowStart;
			// Subtracting the row's largest score first means the biggest
			// value passed to exp is 0, so nothing can overflow. It does not
			// change the result, because it scales the top and the bottom of
			// every fraction by the same amount.
			const double maxScore = *std::max_element(rowBegin, rowBegin + width);
			for (std::size_t col = 0; col < width; col++) {
				probabilities[rowStart + col] = std::exp(scoreData[rowStart + col] - maxScore);
			}
			const double rowTotal = std::accumulate(probabilities.begin() + rowStart,
				probabilities.begin() + rowStart + width, 0.0);
			for (std::size_t col = 0; col < width; col++) {
				probabilities[rowStart + col] /= rowTotal;
			}
		}
		return Tensor(scores.shape(), std::move(probabilities));
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

	// Slope is 1 for positive inputs and 0 otherwise. At exactly zero there is
	// no single answer; 0 is the common choice, and PyTorch uses it too.
	Tensor reluDerivitive(const Tensor& tensor) {
		const auto& input_data = tensor.getData();
		std::vector<double> grad(input_data.size());
		for (std::size_t i = 0; i < input_data.size(); i++) {
			grad[i] = input_data[i] > 0 ? 1.0 : 0.0;
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

	Tensor softMax(const Tensor& scores) {
		double maxScore = *std::max_element(scores.getData().begin(), scores.getData().end());
		std::vector<double> probabilities(scores.getData().size());
		for (std::size_t i = 0; i < scores.getData().size(); i++) {
			probabilities[i] = std::exp(scores.getData()[i] - maxScore);
		}
		double sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
		for (std::size_t i = 0; i < probabilities.size(); i++) {
			probabilities[i] /= sum;
		}
		return Tensor(scores.shape(), std::move(probabilities));
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

	// MULTI-CLASS PREDICTION FEATURES

	namespace {
		// Both functions below rely on targets shaped like the scores, and on
		// every target row adding up to 1. The gradient formula needs the
		// second, and a broken one-hot row (all zeros, say) would otherwise
		// train silently on a loss of zero.
		void checkSoftmaxCrossEntropyInputs(const Tensor& scores, const Tensor& targets) {
			if (scores.shape() != targets.shape()) {
				throw std::invalid_argument("Scores and targets must have the same shape");
			}
			const std::vector<double>& targetData = targets.getData();
			const std::size_t width = targets.shape().back();
			for (std::size_t rowStart = 0; rowStart < targetData.size(); rowStart += width) {
				const double rowTotal = std::accumulate(targetData.begin() + rowStart,
					targetData.begin() + rowStart + width, 0.0);
				if (std::abs(rowTotal - 1.0) > 1.0e-9) {
					throw std::invalid_argument("Each target row must add up to 1");
				}
			}
		}
	}

	double softmaxCrossEntropy(const Tensor& scores, const Tensor& targets) {
		checkSoftmaxCrossEntropyInputs(scores, targets);
		const std::vector<double>& scoreData = scores.getData();
		const std::vector<double>& targetData = targets.getData();
		const std::size_t width = scores.shape().back();
		const std::size_t rows = scores.numel() / width;

		double totalLoss = 0.0;
		for (std::size_t row = 0; row < rows; row++) {
			const std::size_t rowStart = row * width;
			const auto rowBegin = scoreData.begin() + rowStart;
			const double maxScore = *std::max_element(rowBegin, rowBegin + width);
			double rowTotal = 0.0;
			for (std::size_t col = 0; col < width; col++) {
				rowTotal += std::exp(scoreData[rowStart + col] - maxScore);
			}
			// The log of each probability, worked out without ever forming the
			// probability itself. A probability too small for a double would
			// round to 0, and log(0) is minus infinity; this stays finite.
			const double logRowTotal = std::log(rowTotal);
			for (std::size_t col = 0; col < width; col++) {
				const double logProbability = scoreData[rowStart + col] - maxScore - logRowTotal;
				totalLoss -= targetData[rowStart + col] * logProbability;
			}
		}
		return totalLoss / static_cast<double>(rows);
	}

	// Softmax and the loss together have a simple gradient: probabilities
	// minus targets. The loss averages over rows, so each row's share is
	// divided by the row count.
	Tensor softmaxCrossEntropyGrad(const Tensor& scores, const Tensor& targets) {
		checkSoftmaxCrossEntropyInputs(scores, targets);
		const std::size_t rows = scores.numel() / scores.shape().back();
		return (softmax(scores) - targets) * (1.0 / static_cast<double>(rows));
	}


	Tensor createBias(std::size_t layerCount) {
		return zeros(Tensor::Shape({layerCount}));
	}

	// He initialization: every weight is drawn evenly from -limit to +limit,
	Tensor createWeight(const size_t numInputs, const size_t numOutputs, std::mt19937& generator) {
		// Checked before the limit is computed, which would divide by zero.
		if (numInputs == 0 || numOutputs == 0) {
			throw std::invalid_argument("createWeight needs at least one input and one output");
		}
		const double limit = std::sqrt(6.0 / static_cast<double>(numInputs));
		std::uniform_real_distribution<double> distribution(-limit, limit);
		std::vector<double> weightData(numInputs * numOutputs);
		for (size_t i = 0; i < weightData.size(); ++i) {
			weightData[i] = distribution(generator);
		}
		return Tensor({numInputs, numOutputs}, weightData);
	}

	

}


