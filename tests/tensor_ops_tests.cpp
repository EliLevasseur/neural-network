#include "nnet/core/tensor_ops.h"
#include "test_utils.h"
#include "nnet/core/tensor.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

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

	const double sigNumeric =
		(sigmoid(nnet::Tensor({1}, {0.5 + h})).at({0}) - sigmoid(nnet::Tensor({1}, {0.5 - h})).at({0})) / (2 * h);
	const nnet::Tensor sigInput({1}, {0.5});
	tests.expectNear(sigNumeric, nnet::sigmoidDerivitive(sigInput).at({0}), 1.0e-6,
		"sigmoidGrad matches a central finite difference");

	const nnet::Tensor reluInput({4}, {-2.0, -0.5, 0.3, 1.7});
	tests.expectTrue(nnet::relu(reluInput).getData() == std::vector<double>{0.0, 0.0, 0.3, 1.7},
		"relu keeps positive values and replaces negative values with zero");
	tests.expectTrue(nnet::reluDerivitive(nnet::Tensor({1}, {0.0})).at({0}) == 0.0,
		"reluDerivitive is zero at exactly zero");

	// A finite difference straddling zero would measure 0.5, which matches
	// neither side, so these points stay clear of it.
	const nnet::Tensor reluPoints({4}, {-1.5, -0.3, 0.4, 2.0});
	const nnet::Tensor reluSlopes = nnet::reluDerivitive(reluPoints);
	bool reluSlopesMatch = true;
	for (std::size_t i = 0; i < reluPoints.numel(); ++i) {
		const double point = reluPoints.at({i});
		const double numeric =
			(nnet::relu(nnet::Tensor({1}, {point + h})).at({0}) - nnet::relu(nnet::Tensor({1}, {point - h})).at({0})) / (2 * h);
		if (std::abs(numeric - reluSlopes.at({i})) > 1.0e-6) {
			reluSlopesMatch = false;
		}
	}
	tests.expectTrue(reluSlopesMatch, "reluDerivitive matches a central finite difference away from zero");

	// Weight initialization: every weight is drawn evenly from -limit to
	// +limit, where limit = sqrt(6 / inputs).
	{
		std::mt19937 firstGenerator(7);
		std::mt19937 secondGenerator(7);
		const nnet::Tensor firstWeights = nnet::createWeight(784, 128, firstGenerator);
		const nnet::Tensor secondWeights = nnet::createWeight(784, 128, secondGenerator);
		tests.expectTrue(firstWeights.shape() == nnet::Tensor::Shape{784, 128},
			"createWeight makes an inputs by outputs weight");
		tests.expectTrue(firstWeights.getData() == secondWeights.getData(),
			"createWeight gives identical weights from identical seeds");

		std::mt19937 otherSeed(8);
		tests.expectTrue(nnet::createWeight(784, 128, otherSeed).getData() != firstWeights.getData(),
			"createWeight gives different weights from a different seed");
		tests.expectTrue(nnet::createWeight(784, 128, firstGenerator).getData() != firstWeights.getData(),
			"the next draw from the same generator gives different weights");

		const double limit = std::sqrt(6.0 / 784.0);
		const auto [smallest, largest] =
			std::minmax_element(firstWeights.getData().begin(), firstWeights.getData().end());
		tests.expectTrue(*smallest >= -limit && *largest <= limit,
			"every weight stays inside plus or minus sqrt(6 / inputs)");
		// 100352 draws land very close to both ends of the range. This catches
		// a range that is too narrow, or one that only covers one side of zero.
		tests.expectTrue(*smallest < -0.99 * limit && *largest > 0.99 * limit,
			"weights fill the whole range on both sides of zero");

		bool rejectedNoInputs = false;
		try {
			nnet::createWeight(0, 3, firstGenerator);
		} catch (const std::invalid_argument&) {
			rejectedNoInputs = true;
		}
		tests.expectTrue(rejectedNoInputs, "createWeight rejects a layer with no inputs");
	}

	// Softmax and multi-class cross-entropy. The expected values are worked
	// out here straight from the definition, without the largest-score shift.
	{
		const nnet::Tensor scores({2, 3}, {2.0, 1.0, 0.1, 0.5, 0.5, 3.0});
		const nnet::Tensor targets({2, 3}, {1.0, 0.0, 0.0, 0.0, 0.0, 1.0});

		const nnet::Tensor probabilities = nnet::softmax(scores);
		bool softmaxMatches = true;
		for (std::size_t row = 0; row < 2; ++row) {
			double rowTotal = 0.0;
			for (std::size_t col = 0; col < 3; ++col) {
				rowTotal += std::exp(scores.at({row, col}));
			}
			for (std::size_t col = 0; col < 3; ++col) {
				const double expected = std::exp(scores.at({row, col})) / rowTotal;
				if (!(std::abs(probabilities.at({row, col}) - expected) <= tolerance)) {
					softmaxMatches = false;
				}
			}
		}
		tests.expectTrue(softmaxMatches, "softmax matches the definition separately on every row of a batch");

		// e^1000 overflows a double, so without the shift these would be NaN.
		const nnet::Tensor hugeProbabilities = nnet::softmax(nnet::Tensor({1, 3}, {1000.0, 999.0, 998.0}));
		const nnet::Tensor smallProbabilities = nnet::softmax(nnet::Tensor({1, 3}, {2.0, 1.0, 0.0}));
		bool hugeMatches = true;
		for (std::size_t col = 0; col < 3; ++col) {
			if (!(std::abs(hugeProbabilities.at({0, col}) - smallProbabilities.at({0, col})) <= tolerance)) {
				hugeMatches = false;
			}
		}
		tests.expectTrue(hugeMatches, "softmax of scores near 1000 matches the same scores shifted down");

		const double expectedLoss =
			-(std::log(std::exp(2.0) / (std::exp(2.0) + std::exp(1.0) + std::exp(0.1))) +
			  std::log(std::exp(3.0) / (std::exp(0.5) + std::exp(0.5) + std::exp(3.0)))) / 2.0;
		tests.expectNear(nnet::softmaxCrossEntropy(scores, targets), expectedLoss, tolerance,
			"softmaxCrossEntropy averages minus the log of the correct probability over rows");

		// The correct class gets a probability of about e^-1000, which rounds
		// to 0 in a double, and log(0) is minus infinity. Working in logs
		// gives the true loss, 1000, and a finite gradient.
		const nnet::Tensor confidentAndWrong({1, 2}, {1000.0, 0.0});
		const nnet::Tensor secondIsCorrect({1, 2}, {0.0, 1.0});
		tests.expectNear(nnet::softmaxCrossEntropy(confidentAndWrong, secondIsCorrect), 1000.0, 1.0e-9,
			"softmaxCrossEntropy stays finite when the correct class has a vanishing probability");
		const nnet::Tensor wrongGradient = nnet::softmaxCrossEntropyGrad(confidentAndWrong, secondIsCorrect);
		tests.expectNear(wrongGradient.at({0, 0}), 1.0, tolerance,
			"the confidently wrong score is pushed down at full strength");
		tests.expectNear(wrongGradient.at({0, 1}), -1.0, tolerance,
			"the neglected correct score is pushed up at full strength");

		const nnet::Tensor analyticGradient = nnet::softmaxCrossEntropyGrad(scores, targets);
		bool gradientMatches = true;
		for (std::size_t i = 0; i < scores.numel(); ++i) {
			std::vector<double> plusData = scores.getData();
			std::vector<double> minusData = scores.getData();
			plusData[i] += h;
			minusData[i] -= h;
			const double numeric =
				(nnet::softmaxCrossEntropy(nnet::Tensor({2, 3}, plusData), targets) -
				 nnet::softmaxCrossEntropy(nnet::Tensor({2, 3}, minusData), targets)) / (2 * h);
			if (!(std::abs(numeric - analyticGradient.getData()[i]) <= 1.0e-6)) {
				gradientMatches = false;
			}
		}
		tests.expectTrue(gradientMatches, "softmaxCrossEntropyGrad matches a central finite difference");

		bool rejectedShape = false;
		try {
			nnet::softmaxCrossEntropy(scores, nnet::Tensor({3, 2}, {1.0, 0.0, 0.0, 1.0, 1.0, 0.0}));
		} catch (const std::invalid_argument&) {
			rejectedShape = true;
		}
		tests.expectTrue(rejectedShape, "softmaxCrossEntropy rejects targets shaped differently from the scores");

		bool rejectedRow = false;
		try {
			nnet::softmaxCrossEntropy(scores, nnet::Tensor({2, 3}, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
		} catch (const std::invalid_argument&) {
			rejectedRow = true;
		}
		tests.expectTrue(rejectedRow, "softmaxCrossEntropy rejects a target row that does not add up to 1");
	}

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
	tests.expectTrue(transposeRejectedRank, "transpose rejects a tensor below rank 2");

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
	
	
	
	
	
		
	// ---- rank 3: leading dimensions are batch dimensions ----------------
	// A stack of two matrices, each 2 rows by 3 columns.
	const nnet::Tensor batched({2, 2, 3}, {1, 2, 3,  4, 5, 6,
	                                       7, 8, 9, 10, 11, 12});

	const nnet::Tensor batchedTransposed = nnet::transpose(batched);
	tests.expectTrue(batchedTransposed.shape() == nnet::Tensor::Shape{2, 3, 2} &&
		batchedTransposed.getData() == std::vector<double>{1, 4, 2, 5, 3, 6,
		                                                   7, 10, 8, 11, 9, 12},
		"transpose swaps the last two axes and leaves the batch axis alone");

	nnet::Tensor batchedWithBias = batched;
	nnet::addBias(batchedWithBias, nnet::Tensor({3}, {100, 200, 300}));
	tests.expectTrue(batchedWithBias.getData() == std::vector<double>{
			101, 202, 303, 104, 205, 306,
			107, 208, 309, 110, 211, 312},
		"addBias adds to the last axis of every row at any rank");

	const nnet::Tensor batchedBiasGrad =
		nnet::addBiasGrad(batched, nnet::Tensor({3}, {0, 0, 0}));
	tests.expectTrue(batchedBiasGrad.shape() == nnet::Tensor::Shape{3} &&
		batchedBiasGrad.getData() == std::vector<double>{22, 26, 30},
		"addBiasGrad sums over every leading dimension");

	const nnet::Tensor batchedUpstream({2, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 1});
	const nnet::Tensor batchedWeightGrad =
		nnet::matmulGradWeight(batched, batchedUpstream);
	tests.expectTrue(batchedWeightGrad.shape() == nnet::Tensor::Shape{2, 3, 2} &&
		batchedWeightGrad.getData() == std::vector<double>{5, 5, 7, 7, 9, 9,
		                                                   17, 17, 19, 19, 21, 21},
		"matmulGradWeight works on a batch, inheriting the fix from transpose");

	// A rank-1 bias add now works too, which the rank-2 version could not do.
	nnet::Tensor biasedVector({3}, {1, 2, 3});
	nnet::addBias(biasedVector, nnet::Tensor({3}, {10, 20, 30}));
	tests.expectTrue(biasedVector.getData() == std::vector<double>{11, 22, 33},
		"addBias works on a rank-1 tensor");

	// ---- one shared weight against a whole batch ------------------------
	// Three levels of data: 2 groups, 2 rows each, 3 numbers per row. Every
	// group must go through the SAME weight grid, not one grid each.
	const nnet::Tensor grouped({2, 2, 3}, {1, 2, 3,  4, 5, 6,
	                                       7, 8, 9, 10, 11, 12});
	const nnet::Tensor sharedWeight({3, 2}, {1, 0,
	                                         0, 1,
	                                         1, 1});
	const nnet::Tensor sharedProduct = grouped * sharedWeight;
	tests.expectTrue(sharedProduct.shape() == nnet::Tensor::Shape{2, 2, 2} &&
		sharedProduct.getData() == std::vector<double>{4, 5, 10, 11, 16, 17, 22, 23},
		"matmul reuses a single rank-2 right operand for every batch");

	bool matmulRejectedMiddleRank = false;
	try {
		(void)(grouped * nnet::Tensor({2, 3, 2, 1}, {1, 2, 3, 4, 5, 6, 7, 8,
		                                             9, 10, 11, 12}));
	} catch (const std::invalid_argument&) {
		matmulRejectedMiddleRank = true;
	}
	tests.expectTrue(matmulRejectedMiddleRank,
		"matmul still rejects a right operand that is neither matching nor rank 2");

	// ---- summing a shared value's gradient contributions ----------------
	const nnet::Tensor summedDown = nnet::sumLeadingDimensions(grouped, 2);
	tests.expectTrue(summedDown.shape() == nnet::Tensor::Shape{2, 3} &&
		summedDown.getData() == std::vector<double>{8, 10, 12, 14, 16, 18},
		"sumLeadingDimensions adds the leading axis away");

	tests.expectTrue(
		nnet::sumLeadingDimensions(grouped, 3).getData() == grouped.getData(),
		"sumLeadingDimensions leaves a tensor alone when the rank already matches");

}
