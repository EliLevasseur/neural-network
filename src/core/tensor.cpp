#include "../../include/nnet/core/tensor.h"

namespace nnet {

Tensor::Tensor(Tensor::Shape shape, std::vector<double> values) : shape_(std::move(shape)), data_(std::move(values)) {
	if (shape_.size() <= 1)
		throw std::invalid_arguement("Shape must be at least one dimension");

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

const Tensor::Shape& Tensor::strides() const {
	return strides_;
	}


}	

