#include "nnet/core/tensor_ops.h"
#include <cmath>

namespace nnet {

	Tensor sigmoid(const Tensor& tensor) {
		const auto& input_data = tensor.getData();
		std::vector<double> out(input_data.size());
		for (size_t i = 0; i < input_data.size(); ++i) {
			out[i] = 1 / (1 + std::exp(-input_data[i]));
		}
		return Tensor(tensor.shape(), out);
	}

	double sum(const Tensor& tensor) {
		const auto& input_data = tensor.getData();
		double total = 0.0;
		for (const auto& value : input_data) {
			total += value;
		}
		return total;
	}

}
