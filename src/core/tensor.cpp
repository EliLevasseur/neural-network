#include "../../include/nnet/core/tensor.h"

namespace nnet {

Tensor::Tensor(Tensor::Shape shape, std::vector<double> values) : shape_(std::move(shape)), data_(std::move(values)) {

	if (shape_.size() < 1)
		throw std::invalid_argument("Shape must be at least one dimension");
	
	std::size_t sizeTest = 1;

	for (auto& val : shape_) {
		if (val == 0)
			throw std::invalid_argument("Cannot create a shape with negative dimensions");
		sizeTest *= val;
	}
	
	if (sizeTest != data_.size()) {
		throw std::invalid_argument("Data must contain (rows * columns) elements");
	}

	for (std::size_t i = 0; i < shape_.size() - 1; i++) {
		std::size_t temp = 1;
		for (std::size_t j = i + 1; j < shape_.size(); j++) {
				temp *= shape_[j];
				}
		strides_.push_back(temp);
	}
	strides_.push_back(1);
	}

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

std::size_t Tensor::calculateOffset(const Shape& indices) const {
	std::size_t offset = 0;
	for (std::size_t i = 0; i < indices.size(); i++) {
		if (indices[i] >= shape_[i])
			throw std::out_of_range("Tensor index out of range");
		offset += indices[i] * strides_[i];
		}
	return offset;
	}


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

// OPERATOR OVERLOADERS
Tensor Tensor::operator+(const Tensor& other) const {
	if (shape_ != other.shape())
		throw std::invalid_argument("Tensors must be of same size for addition");
	
	std::vector<double> values(data_.size());

	for (std::size_t i = 0; i < data_.size(); i++) {
		values[i] = data_[i] + other.data_[i];
		}
	return Tensor(shape_, values);
	}

Tensor Tensor::operator*(double scalar) const {
	std::vector<double> values(data_.size());

	for (std::size_t i = 0; i < data_.size(); i++) {
		values[i] = data_[i] * scalar;
	}
	return Tensor(shape_, values);
}

Tensor Tensor::operator-(const Tensor& other) const {
	if (shape_ != other.shape())
		throw std::invalid_argument("Tensors must be of same size for subtraction");
	
	std::vector<double> values(data_.size());

	for (std::size_t i = 0; i < data_.size(); i++) {
		values[i] = data_[i] - other.data_[i];
		}
	return Tensor(shape_, values);
	}
}
