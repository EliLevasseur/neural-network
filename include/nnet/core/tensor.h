#ifndef TENSOR_TEST_H
#define TENSOR_TEST_H

#include <vector>
#include <stdexcept>

namespace nnet {


class Tensor {	

	public:
		// This the the n * m * ... shape  of a tensor
		using Shape = std::vector<std::size_t>;	

		Tensor(Shape shape, std::vector<double> values);
		std::size_t rank() const;
		std::size_t numel() const;

		const std::vector<double>& getData() const;

		const Shape& strides() const;
		const Shape& shape() const;
		
		void inplaceMultiplication(const Tensor& otherTensor);
		
		// Methods
		double& at(const Shape& indices);
		const double& at(const Shape& indices) const;

		// Explicit default decleration of copy and move constructors and assignment operators
		Tensor(const Tensor& other) = default;
		Tensor(Tensor&& other) = default;
		Tensor& operator=(const Tensor& other) = default;
		Tensor& operator=(Tensor&& other) = default;
		
		// Operator overloaders
		Tensor operator+(const Tensor& other) const;
		Tensor operator*(double scalar) const;
		Tensor operator*(const Tensor& other) const;
		Tensor operator-(const Tensor& other) const;

	private:
		Shape shape_;
		Shape strides_;
		std::vector<double> data_;
		
		std::size_t calculateOffset(const Shape& indices) const;	

	};
}

#endif
