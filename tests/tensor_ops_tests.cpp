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
}
