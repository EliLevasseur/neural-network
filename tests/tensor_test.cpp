#include "test_utils.h"
#include "nnet/core/tensor.h"


void runTensorTests(TestRunner& tests) {

	tests.section("TENSOR TESTS");

	std::vector<std::size_t> shape{3, 2};

	std::vector<double> data{2, 4, 5, 4, 2, 1};
	std::vector<double> data2{1, 1, 1, 1, 1, 1};
	std::vector<double> valid_add_data{3, 5, 6, 5, 3, 2};
	std::vector<double> valid_minus_data{1, 3, 4, 3, 1, 0};
	std::vector<double> valid_scaled_data{8, 16, 20, 16, 8, 4};


	nnet::Tensor tensor(shape, data);
	nnet::Tensor tensor2(shape, data2);


	std::size_t numel = tensor.numel();
	std::size_t rank = tensor.rank();

	nnet::Tensor::Shape strides = tensor.strides();
	
	nnet::Tensor sum = tensor + tensor2;
	nnet::Tensor difference = tensor - tensor2;
	nnet::Tensor scaledTensor = tensor * 4;
	
	// gradient checker (VJP Verification)
	std::vector<double> xData{2.0, 5.0};
	std::vector<double> yData{3.0, 4.0};
	std::vector<double> upstream{10.0, 20.0};
	
	// test overloaders
	tests.expectTrue(difference.getData() == valid_minus_data, "subtraction operator overload works");
	tests.expectTrue(sum.getData() == valid_add_data, "addition operator overload works");
	tests.expectTrue(scaledTensor.getData() == valid_scaled_data, "scalar multiplication operator overload works");

	// test class methods
	tests.expectTrue(strides == std::vector<size_t>{2, 1}, "strides(shape{3, 2}) --> {2, 1}");
	tests.expectTrue(numel == 6, "numel(data{3, 2}) --> 6");
	tests.expectTrue(rank == 2, "rank(shape{3, 2}) --> 2");
	tests.expectTrue(tensor.at({2, 1}) == 1, "at() calls 1 at index {2, 1}");
	tests.expectTrue(tensor.shape() == nnet::Tensor::Shape{3, 2}, "shape == {3, 2}");
}
