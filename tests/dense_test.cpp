#include "test_utils.h"
#include "nnet/nn/dense.h"

void runDenseTests(TestRunner& tests) {
	tests.section("DENSE LAYER TESTS");

	// ---- construction ---------------------------------------------------
	// Both shapes come from the same two numbers, so a weight sized for one
	// layer can never sit next to a bias sized for another.
	nnet::Dense layer(3, 2);

	tests.expectTrue(layer.weight.value.shape() == nnet::Tensor::Shape{3, 2},
		"Dense(3, 2) creates a weight shaped inputs by outputs");

	tests.expectTrue(layer.bias.value.shape() == nnet::Tensor::Shape{2},
		"Dense(3, 2) creates one bias value per output");

	// A gradient is born matching the value it belongs to, and starts at
	// zero so it can be read before anything has written to it.
	tests.expectTrue(layer.weight.grad.shape() == layer.weight.value.shape() &&
		layer.bias.grad.shape() == layer.bias.value.shape(),
		"each gradient is born with the same shape as its value");

	tests.expectTrue(layer.weight.grad.getData() == std::vector<double>(6, 0.0) &&
		layer.bias.grad.getData() == std::vector<double>(2, 0.0),
		"gradients start filled with zero");

	// ---- parameters() ---------------------------------------------------
	const std::vector<nnet::Parameter*> collected = layer.parameters();

	tests.expectTrue(collected.size() == 2,
		"parameters() reports both trainable values of the layer");

	tests.expectTrue(collected[0] == &layer.weight && collected[1] == &layer.bias,
		"parameters() points at the layer's own members, not at copies");

	// The point of handing out pointers is that an optimizer can change the
	// real weights. Writing through the pointer must be visible in the layer.
	collected[0]->value.at({0, 0}) = 99.0;
	tests.expectNear(layer.weight.value.at({0, 0}), 99.0, 1.0e-12,
		"writing through a parameter pointer changes the layer itself");

	// ---- forward --------------------------------------------------------
	// Weights are randomly initialised, so overwrite them with known values
	// to get a result that can be checked by hand.
	nnet::Dense knownLayer(2, 2);
	knownLayer.weight.value = nnet::Tensor({2, 2}, {1, 2,
	                                                3, 4});
	knownLayer.bias.value = nnet::Tensor({2}, {10, 20});

	const nnet::Tensor input({1, 2}, {1, 1});
	const nnet::Tensor output = knownLayer.forward(input);

	// First output  = 1*1 + 1*3 = 4, plus bias 10 = 14
	// Second output = 1*2 + 1*4 = 6, plus bias 20 = 26
	tests.expectTrue(output.shape() == nnet::Tensor::Shape{1, 2},
		"forward returns one row with one value per output");

	tests.expectNear(output.at({0, 0}), 14.0, 1.0e-12,
		"forward computes the first output by hand-checked arithmetic");

	tests.expectNear(output.at({0, 1}), 26.0, 1.0e-12,
		"forward computes the second output by hand-checked arithmetic");

	// ---- forward keeps no state ------------------------------------------
	// A layer that cached its last input would return something different
	// the second time, and would compute a wrong gradient when reused.
	const nnet::Tensor secondOutput = knownLayer.forward(input);
	tests.expectTrue(secondOutput.getData() == output.getData(),
		"calling forward twice with the same input gives the same answer");

	const nnet::Tensor otherInput({1, 2}, {2, 0});
	const nnet::Tensor otherOutput = knownLayer.forward(otherInput);
	tests.expectNear(otherOutput.at({0, 0}), 12.0, 1.0e-12,
		"a later call is unaffected by an earlier one");

	tests.expectTrue(knownLayer.weight.value.at({0, 0}) == 1.0 &&
		knownLayer.bias.value.at({0}) == 10.0,
		"forward leaves the layer's own weight and bias untouched");
	// ---- a layer applied across grouped (rank-3) data -------------------
	// 2 groups, 2 rows each, 3 numbers per row, through one Dense(3, 2).
	nnet::Dense groupedLayer(3, 2);
	groupedLayer.weight.value = nnet::Tensor({3, 2}, {1, 0,
	                                                  0, 1,
	                                                  1, 1});
	groupedLayer.bias.value = nnet::Tensor({2}, {0, 0});

	const nnet::Tensor groupedInput({2, 2, 3}, {1, 2, 3,  4, 5, 6,
	                                            7, 8, 9, 10, 11, 12});
	const nnet::Tensor groupedOutput = groupedLayer.forward(groupedInput);

	tests.expectTrue(groupedOutput.shape() == nnet::Tensor::Shape{2, 2, 2} &&
		groupedOutput.getData() == std::vector<double>{4, 5, 10, 11, 16, 17, 22, 23},
		"one layer runs across grouped data using a single shared weight");

	// Backward must fold every group's contribution back into the one weight,
	// so the gradient comes out shaped like the weight rather than the batch.
	const nnet::Tensor groupedUpstream({2, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 1});
	groupedLayer.backward(groupedInput, groupedUpstream);

	tests.expectTrue(groupedLayer.weight.grad.shape() == nnet::Tensor::Shape{3, 2},
		"a shared weight's gradient has the weight's shape, not the batch's");

	// Independent check: do each group separately and add the two results.
	const nnet::Tensor firstGroup({2, 3}, {1, 2, 3, 4, 5, 6});
	const nnet::Tensor secondGroup({2, 3}, {7, 8, 9, 10, 11, 12});
	const nnet::Tensor onesRank2({2, 2}, {1, 1, 1, 1});
	const nnet::Tensor byHand =
		nnet::matmulGradWeight(firstGroup, onesRank2) +
		nnet::matmulGradWeight(secondGroup, onesRank2);

	tests.expectTrue(groupedLayer.weight.grad.getData() == byHand.getData(),
		"the grouped weight gradient equals each group done separately and added");

	tests.expectTrue(groupedLayer.bias.grad.shape() == nnet::Tensor::Shape{2} &&
		groupedLayer.bias.grad.getData() == std::vector<double>{4, 4},
		"the shared bias gradient sums over every group and row");

}
