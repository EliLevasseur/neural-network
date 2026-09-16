#include "nnet/core/tensor_ops.h"
#include "test_utils.h"
#include "nnet/core/tensor.h"

void runOperationTests(TestRunner& tests) {
	tests.section("TENSOR OPERATION TESTS");

	nnet::Tensor::Shape shape{2, 3};
	std::vector<double> data{2, 1, 3, 2, 1, 2};
	
	nnet::Tensor tensor(shape, data);

	const double tolerance = 1.0e-12;
	tests.expectNear(1.0 / (1.0 + std::exp(-(tensor.at({1, 0})))), sigmoid(tensor).at({1, 0}), tolerance, "Sigmoid works");

	const nnet::Tensor filled = nnet::fill(nnet::Tensor::Shape{2, 2}, 7.0);
	tests.expectTrue(filled.getData() == std::vector<double>{7, 7, 7, 7} &&
		filled.shape() == nnet::Tensor::Shape{2, 2},
		"fill() creates a tensor of the given shape filled with the given value");

	const nnet::Tensor zeroed = nnet::zeros(nnet::Tensor::Shape{3});
	tests.expectTrue(zeroed.getData() == std::vector<double>{0, 0, 0},
		"zeros() creates a tensor of the given shape filled with zero");

	const nnet::Tensor probabilities({2}, {0.9, 0.2});
	const nnet::Tensor labels({2}, {1.0, 0.0});
	const double expectedLoss = -(std::log(0.9) + std::log(0.8)) / 2.0;
	tests.expectNear(expectedLoss, nnet::binaryCrossEntropy(probabilities, labels), tolerance,
		"binaryCrossEntropy matches a hand-computed two-sample loss");

	bool bceRejectedMismatch = false;
	try {
		nnet::binaryCrossEntropy(probabilities, nnet::Tensor({3}, {1.0, 0.0, 1.0}));
	} catch (const std::invalid_argument&) {
		bceRejectedMismatch = true;
	}
	tests.expectTrue(bceRejectedMismatch, "binaryCrossEntropy rejects mismatched shapes");

	const nnet::Tensor analyticGrad = nnet::binaryCrossEntropyGrad(probabilities, labels);
	const double h = 1.0e-6;
	const nnet::Tensor predsPlus({2}, {0.9 + h, 0.2});
	const nnet::Tensor predsMinus({2}, {0.9 - h, 0.2});
	const double numericGrad0 =
		(nnet::binaryCrossEntropy(predsPlus, labels) - nnet::binaryCrossEntropy(predsMinus, labels)) / (2 * h);
	tests.expectNear(numericGrad0, analyticGrad.at({0}), 1.0e-6,
		"binaryCrossEntropyGrad matches a central finite difference at an interior point");

	const nnet::Tensor sigInput({1}, {0.5});
	const double sigNumeric =
		(sigmoid(nnet::Tensor({1}, {0.5 + h})).at({0}) - sigmoid(nnet::Tensor({1}, {0.5 - h})).at({0})) / (2 * h);
	tests.expectNear(sigNumeric, nnet::sigmoidGrad(sigInput).at({0}), 1.0e-6,
		"sigmoidGrad matches a central finite difference");

	const nnet::Tensor matrixForBiasGrad({2, 2}, {1, 2, 3, 4});
	const nnet::Tensor upstreamOnes({2, 2}, {1, 1, 1, 1});
	const nnet::Tensor biasGradAnalytic = nnet::addBiasGrad(upstreamOnes, nnet::Tensor({2}, {0.1, 0.2}));

	auto sumAfterBias = [](nnet::Tensor matrix, const nnet::Tensor& bias) {
		nnet::addBias(matrix, bias);
		return nnet::sum(matrix);
	};
	const double biasGradNumeric0 =
		(sumAfterBias(matrixForBiasGrad, nnet::Tensor({2}, {0.1 + h, 0.2})) -
		 sumAfterBias(matrixForBiasGrad, nnet::Tensor({2}, {0.1 - h, 0.2}))) / (2 * h);
	tests.expectNear(biasGradNumeric0, biasGradAnalytic.at({0}), 1.0e-6,
		"addBiasGrad matches a central finite difference");

	const nnet::Tensor transposed = nnet::transpose(nnet::Tensor({2, 3}, {1, 2, 3, 4, 5, 6}));
	tests.expectTrue(transposed.shape() == nnet::Tensor::Shape{3, 2} &&
		transposed.getData() == std::vector<double>{1, 4, 2, 5, 3, 6},
		"transpose swaps rows and columns");

	bool transposeRejectedRank = false;
	try {
		nnet::transpose(nnet::Tensor({3}, {1, 2, 3}));
	} catch (const std::invalid_argument&) {
		transposeRejectedRank = true;
	}
	tests.expectTrue(transposeRejectedRank, "transpose rejects a non-rank-2 tensor");

	const nnet::Tensor x({2, 2}, {1, 2, 3, 4});
	const nnet::Tensor w({2, 2}, {5, 6, 7, 8});
	const nnet::Tensor upstreamOnes2x2({2, 2}, {1, 1, 1, 1});

	auto sumOfMatmul = [](const nnet::Tensor& left, const nnet::Tensor& right) {
		return nnet::sum(left * right);
	};

	const nnet::Tensor dW = nnet::matmulGradWeight(x, upstreamOnes2x2);
	const double dW00Numeric =
		(sumOfMatmul(x, nnet::Tensor({2, 2}, {5 + h, 6, 7, 8})) -
		 sumOfMatmul(x, nnet::Tensor({2, 2}, {5 - h, 6, 7, 8}))) / (2 * h);
	tests.expectNear(dW00Numeric, dW.at({0, 0}), 1.0e-6,
		"matmulGradWeight matches a central finite difference");

	const nnet::Tensor dX = nnet::matmulGradInput(upstreamOnes2x2, w);
	const double dX00Numeric =
		(sumOfMatmul(nnet::Tensor({2, 2}, {1 + h, 2, 3, 4}), w) -
		 sumOfMatmul(nnet::Tensor({2, 2}, {1 - h, 2, 3, 4}), w)) / (2 * h);
	tests.expectNear(dX00Numeric, dX.at({0, 0}), 1.0e-6,
		"matmulGradInput matches a central finite difference");
		
	nnet::Tensor X({1, 1}, {2});
	nnet::Tensor W({1, 1}, {0});
	nnet::Tensor Target({1, 1}, {1});
	nnet::Tensor b({1}, {0});
	
	nnet::Tensor combined = X * W;
	addBias(combined, b);
	nnet::Tensor predictions = sigmoid(combined);
	
	double loss = binaryCrossEntropy(predictions, Target);
	tests.expectNear(0.5, predictions.at({0, 0}), 1.0e-6,
		"matmul + sigmoid work");
		
	tests.expectNear(loss, 0.69314718056, 1.0e-6,
		"Binary crossentropy calulates the correct loss");
		
	nnet::Tensor binaryGrad = binaryCrossEntropyGrad(predictions, Target);
	tests.expectTrue(binaryGrad.shape() == nnet::Tensor::Shape{1, 1} && binaryGrad.at({0, 0}) == -2,
	"the binaryCrossEntropygrad returns a correct gradient");
	
	
	
	
	
		
}
