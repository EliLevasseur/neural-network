#include "optim/sgd.h"

namespace nnet {

	void sgdOptimizer(std::vector<Parameter*>& parameters, double learningRate) {
		for (auto& parameter : parameters) {
			parameter->value = parameter->value - parameter->grad * learningRate;
		}
	}
}
