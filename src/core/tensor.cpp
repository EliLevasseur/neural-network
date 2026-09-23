#include "../../include/nnet/core/tensor.h"
#include <cstdint>
#include <utility>

// The Tensor does not currently have thourough testing for indexing COME BACK LATER


namespace nnet {

Tensor::Tensor(Tensor::Shape shape, std::vector<double> values) : shape_(std::move(shape)), data_(std::move(values)) {

	if (shape_.size() < 1)
		throw std::invalid_argument("Shape must be at least one dimension");
	
	std::size_t sizeTest = 1;

	for (auto& val : shape_) {
		if (val == 0)
			throw std::invalid_argument("Cannot create a shape with negative dimensions");
		if (sizeTest > SIZE_MAX / val)
			throw std::overflow_error("Tensor shape is too large to represent");
		sizeTest *= val;
	}
	
	if (sizeTest != data_.size()) {
		throw std::invalid_argument("Data must contain (rows * columns) elements");
	}

	for (std::size_t i = 0; i < shape_.size() - 1; i++) {
		std::size_t temp = 1;
		for (std::size_t j = i + 1; j < shape_.size(); j++) {
				if (temp > SIZE_MAX / shape_[j])
					throw std::overflow_error("Tensor stride is too large to represent");
				temp *= shape_[j];
}
		strides_.push_back(temp);
	}
	strides_.push_back(1);
	}

// ++++++++++++++++ GETTERS ++++++++++++++++++

std::size_t Tensor::rank() const{
	return shape_.size();
	}

std::size_t Tensor::numel() const {
	return data_.size();
	}

const std::vector<double>& Tensor::getData() const {
	return data_;
}

const Tensor::Shape& Tensor::strides() const {
	return strides_;
	}

const Tensor::Shape& Tensor::shape() const {
	return shape_;
	}

//+++++++++++++++++++++++++++++++++++++++++++++++++

// INTERNAL METHODS

std::size_t Tensor::calculateOffset(const Shape& indices) const {
	std::size_t offset = 0;
	for (std::size_t i = 0; i < indices.size(); i++) {
		if (indices[i] >= shape_[i])
			throw std::out_of_range("Tensor index out of range");
		offset += indices[i] * strides_[i];
		}
	return offset;
	}
//----------------------------

double& Tensor::at(const Tensor::Shape& indices) {
	if (indices.size() != shape_.size())
		throw std::invalid_argument("Number of indices must equal tensor dimesions");

	return data_.at(calculateOffset(indices));
	}	

const double& Tensor::at(const Tensor::Shape& indices) const {
	if (indices.size() != shape_.size())
		throw std::invalid_argument("Number of indices must equal tensor dimesions");

	return data_.at(calculateOffset(indices));
	}


// OPERATOR OVERLOADS
Tensor Tensor::operator+(const Tensor& other) const {
	if (shape_ != other.shape())
		throw std::invalid_argument("Tensors must be of same size for addition");

	std::vector<double> values(data_.size());

	for (std::size_t i = 0; i < data_.size(); i++) {
		values[i] = data_[i] + other.data_[i];
	}
	return Tensor(shape_, std::move(values));
}

Tensor Tensor::operator*(double scalar) const {
	std::vector<double> values(data_.size());

	for (std::size_t i = 0; i < data_.size(); i++) {
		values[i] = data_[i] * scalar;
	}
	return Tensor(shape_, std::move(values));
}

// N dimension matrix multiplication baby ;)
Tensor Tensor::operator*(const Tensor& other) const {
    if (rank() < 2 || other.shape_.size() < 2) {
        throw std::invalid_argument(
            "Matrix multiplication requires rank >= 2");
    } 

    // The right operand may either match this tensor's rank, pairing each
    // matrix with its own partner, or be a single rank-2 matrix that every
    // batch shares. The shared case is what a layer applied across grouped
    // data needs: one weight used by every group. Copying that weight once
    // per group instead would break the link between the values the forward
    // pass read and the ones an optimizer later updates.
    if (other.shape_.size() != rank() && other.shape_.size() != 2) {
        throw std::invalid_argument(
            "Right operand must match the left rank or be rank 2");
    }

    if (other.shape_.size() == rank()) {
        for (std::size_t axis = 0; axis < rank() - 2; ++axis) {
            if (shape_[axis] != other.shape_[axis]) {
                throw std::invalid_argument("Batch dimensions must match");
            }
        }
    }

	// Industry standard to represent the last two dimensions as rows and columns for matrix multiplication,
	// preceding values represent batch dimensions.
    const std::size_t rows = shape_[rank() - 2];
    const std::size_t inner = shape_[rank() - 1];
    const std::size_t cols = other.shape_[other.shape_.size() - 1];

    if (inner != other.shape_[other.shape_.size() - 2]) {
        throw std::invalid_argument(
            "Left columns must equal right rows");
    }

    Shape outputShape = shape_;
    outputShape.back() = cols;

	// Each matrix contains rows * inner elements; divide the total to count # of matrices.
    const std::size_t batches = data_.size() / rows / inner;

    // How many matrices the right operand holds. One means it is shared by
    // every batch on the left.
    const std::size_t rightBatches = other.data_.size() / inner / cols;
    std::vector<double> values(batches * rows * cols, 0.0);

    // Multiply each corresponding pair of matrices.
	// Left: rows × inner. Right: inner × cols. Output: rows × cols.
	for (std::size_t batch = 0; batch < batches; ++batch) {

		// Flat position where this batch's matrix begins in each data vector.
		const std::size_t leftStart = batch * rows * inner;
		// A shared right operand never advances: every batch reads the
		// same matrix.
		const std::size_t rightStart =
			(rightBatches == 1) ? 0 : batch * inner * cols;
		const std::size_t outputStart = batch * rows * cols;

		// Visit every position in the output matrix.
		for (std::size_t row = 0; row < rows; ++row) {
			for (std::size_t col = 0; col < cols; ++col) {

				// Accumulate the dot product for this one output element.
				double total = 0.0;

				for (std::size_t k = 0; k < inner; ++k) {
					// Move across the selected row of the left matrix.
					const std::size_t leftIndex = leftStart + row * inner + k;

					// Move down the selected column of the right matrix.
					const std::size_t rightIndex = rightStart + k * cols + col;

					total += data_[leftIndex] * other.data_[rightIndex];
				}

				// Store the completed dot product at output[row, col].
				values[outputStart + row * cols + col] = total;
			}
		}
	}

    return Tensor(std::move(outputShape), std::move(values));
}

Tensor Tensor::operator-(const Tensor& other) const {
	if (shape_ != other.shape())
		throw std::invalid_argument("Tensors must be of same size for subtraction");

	std::vector<double> values(data_.size());

	for (std::size_t i = 0; i < data_.size(); i++) {
		values[i] = data_[i] - other.data_[i];
	}
	return Tensor(shape_, std::move(values));
}

void Tensor::inplaceMultiplication(const Tensor& otherTensor) {

	if (otherTensor.shape_ != shape_)
		throw std::invalid_argument("Tensors must be of the same shape");

	for (size_t val = 0; val < otherTensor.numel(); val++) {
      		data_[val] *= otherTensor.getData()[val];
		}
	}

}



