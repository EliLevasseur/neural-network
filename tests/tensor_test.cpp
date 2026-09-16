#include "test_utils.h"
#include "nnet/core/tensor.h"

#include <utility>


void runTensorTests(TestRunner& tests) {

	tests.section("TENSOR TESTS");

	std::vector<std::size_t> shape{3, 2};
	std::vector<std::size_t> shape3d{2, 2, 2};

	std::vector<double> data{2, 4, 5, 4, 2, 1};
	std::vector<double> data2{1, 1, 1, 1, 1, 1};
	std::vector<double> data3d{1, 2, 1, 2, 1, 2, 1, 2};


	nnet::Tensor tensor(shape, data);
	nnet::Tensor tensor2(shape, data2);
	nnet::Tensor tensor3d(shape3d, data3d); 


	std::size_t numel = tensor.numel();
	std::size_t rank = tensor.rank();

	nnet::Tensor::Shape strides = tensor.strides();
	
	// gradient checker (VJP Verification)
	std::vector<double> xData{2.0, 5.0};
	std::vector<double> yData{3.0, 4.0};
	std::vector<double> upstream{10.0, 20.0};
	

	// test class methods
	tests.expectTrue(strides == std::vector<size_t>{2, 1}, "strides(shape{3, 2}) --> {2, 1}");
	tests.expectTrue(numel == 6, "numel(data{3, 2}) --> 6");
	tests.expectTrue(rank == 2, "rank(shape{3, 2}) --> 2");
	tests.expectTrue(tensor.at({2, 1}) == 1, "at() calls 1 at index {2, 1}");
	tests.expectTrue(tensor.shape() == nnet::Tensor::Shape{3, 2}, "shape == {3, 2}");

	// test 3d shape
	tests.expectTrue(tensor3d.strides() == std::vector<size_t>{4, 2, 1}, "strides(shape{2, 2, 2}) --> {2, 2, 1}");

	tests.section("TENSOR OPERATOR OVERLOADS");

 	// BRUH EVERY TIME I LET AI WRITE THE TESTS THEY ALWAYS BE DOING TOO MUCH {{{{

	// Small exact values make these expected results checkable by hand.
	const auto added = tensor + tensor2;
	const auto subtracted = tensor - tensor2;
	const auto scaled = tensor * -0.5;
	tests.expectTrue(added.getData() == std::vector<double>{3, 5, 6, 5, 3, 2},
		"addition combines corresponding elements");
	tests.expectTrue(subtracted.getData() == std::vector<double>{1, 3, 4, 3, 1, 0},
		"subtraction combines corresponding elements");
	tests.expectTrue(scaled.getData() == std::vector<double>{-1, -2, -2.5, -2, -1, -0.5},
		"scalar multiplication handles a negative fractional factor");
	tests.expectTrue((tensor * 0.0).getData() == std::vector<double>(6, 0.0),
		"scalar multiplication by zero returns zeros");
	tests.expectTrue(added.shape() == shape && subtracted.shape() == shape && scaled.shape() == shape,
		"elementwise operators preserve shape");
	tests.expectTrue(tensor.getData() == data && tensor.shape() == shape &&
		tensor2.getData() == data2 && tensor2.shape() == shape,
		"elementwise operators leave both inputs unchanged");

	// {2,3} * {3,2}: each output is a row-column dot product.
	const nnet::Tensor left({2, 3}, {1, 2, 3, 4, 5, 6});
	const nnet::Tensor right({3, 2}, {10, 20, 30, 40, 50, 60});
	const auto product = left * right;
	tests.expectTrue(product.shape() == nnet::Tensor::Shape{2, 2},
		"matmul returns left rows and right columns");
	tests.expectTrue(product.getData() == std::vector<double>{220, 280, 490, 640},
		"rectangular matmul produces four expected dot products");
	tests.expectTrue(left.shape() == nnet::Tensor::Shape{2, 3} &&
		left.getData() == std::vector<double>{1, 2, 3, 4, 5, 6} &&
		right.shape() == nnet::Tensor::Shape{3, 2} &&
		right.getData() == std::vector<double>{10, 20, 30, 40, 50, 60},
		"matmul leaves both inputs unchanged");

	// The output has six elements, although the left input has only two.
	const nnet::Tensor column({2, 1}, {2, 3});
	const nnet::Tensor row({1, 3}, {4, 5, 6});
	const auto expanded = column * row;
	tests.expectTrue(expanded.shape() == nnet::Tensor::Shape{2, 3},
		"matmul supports an output larger than the left input");
	tests.expectTrue(expanded.getData() == std::vector<double>{8, 10, 12, 12, 15, 18},
		"matmul fills every element of the larger output");

	// Distinct data in each batch catches accidental reuse of the first matrix.
	const nnet::Tensor batchLeft({2, 2, 3},
		{1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6});
	const nnet::Tensor batchRight({2, 3, 2},
		{10, 20, 30, 40, 50, 60, 1, 2, 3, 4, 5, 6});
	const auto batched = batchLeft * batchRight;
	tests.expectTrue(batched.shape() == nnet::Tensor::Shape{2, 2, 2},
		"rank-3 matmul preserves the batch dimension");
	tests.expectTrue(batched.getData() == std::vector<double>{220, 280, 490, 640, -22, -28, -49, -64},
		"rank-3 matmul multiplies corresponding batches independently");

	const nnet::Tensor groupedLeft({2, 2, 1, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
	const nnet::Tensor groupedRight({2, 2, 2, 1}, {1, 0, 0, 1, 1, 1, 2, -1});
	const auto grouped = groupedLeft * groupedRight;
	tests.expectTrue(grouped.shape() == nnet::Tensor::Shape{2, 2, 1, 1},
		"rank-4 matmul preserves both batch dimensions");
	tests.expectTrue(grouped.getData() == std::vector<double>{1, 4, 11, 6},
		"rank-4 matmul visits all four matrix pairs");

	// Keep exception checks local to these operator tests.
	const auto expectInvalidArgument = [&tests](const auto& operation, std::string_view name) {
		bool rejected = false;
		try {
			operation();
		} catch (const std::invalid_argument&) {
			rejected = true;
		}
		tests.expectTrue(rejected, name);
	};

	expectInvalidArgument([&] { (void)(tensor + left); },
		"addition rejects unequal shapes even with equal element counts");
	expectInvalidArgument([&] { (void)(tensor - left); },
		"subtraction rejects unequal shapes even with equal element counts");
	expectInvalidArgument([&] { (void)(left * left); },
		"matmul rejects incompatible inner dimensions");

	const nnet::Tensor vector({3}, {1, 2, 3});
	expectInvalidArgument([&] { (void)(vector * right); },
		"matmul rejects a rank-1 left operand");
	expectInvalidArgument([&] { (void)(left * vector); },
		"matmul rejects a rank-1 right operand");
	expectInvalidArgument([&] { (void)(left * batchRight); },
		"matmul rejects unequal ranks");

	const nnet::Tensor differentGroups({1, 4, 2, 1}, {1, 0, 0, 1, 1, 1, 2, -1});
	expectInvalidArgument([&] { (void)(groupedLeft * differentGroups); },
		"matmul rejects different batch shapes even with equal matrix counts");

	tests.section("TENSOR COPY, MOVE, AND OVERFLOW SAFETY");

	// Copies must be independent: mutating one must never affect the other.
	nnet::Tensor original({2, 2}, {1, 2, 3, 4});
	nnet::Tensor copy = original;
	copy.at({0, 0}) = 99;
	tests.expectTrue(original.at({0, 0}) == 1 && copy.at({0, 0}) == 99,
		"copying a Tensor is independent - mutating the copy leaves the original unchanged");

	// A moved-from Tensor is only guaranteed destructible/assignable - prove
	// the destination got the data, then prove the source works again once
	// something new is assigned into it.
	nnet::Tensor source({2, 2}, {5, 6, 7, 8});
	nnet::Tensor destination = std::move(source);
	tests.expectTrue(destination.getData() == std::vector<double>{5, 6, 7, 8} &&
		destination.shape() == nnet::Tensor::Shape{2, 2},
		"move-construction transfers the original data to the destination");

	source = nnet::Tensor({1}, {42});
	tests.expectTrue(source.numel() == 1 && source.at({0}) == 42,
		"a moved-from Tensor can be reassigned and behaves normally afterward");

	// A shape whose element count cannot fit in size_t must be rejected
	// before the silent wraparound can produce a corrupted Tensor.
	bool overflowRejected = false;
	try {
		nnet::Tensor huge(nnet::Tensor::Shape{5000000000ULL, 5000000000ULL}, std::vector<double>{0.0});
	} catch (const std::overflow_error&) {
		overflowRejected = true;
	}
	tests.expectTrue(overflowRejected,
		"a shape whose element count overflows size_t throws std::overflow_error");

}
// }}}