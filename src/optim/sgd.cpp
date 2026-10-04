#include "optim/sgd.h"

namespace nnet {

	void sgdOptimizer(std::vector<Parameter*>& parameters, double learningRate) {
        for (const auto* parameter : parameters) {
            if (!parameter) {
                throw std::invalid_argument("SGD received a null Parameter");
            }
            parameter->validateShape();
        }
		for (auto& parameter : parameters) {
			if (parameter->requiresGrad) {
				parameter->value = parameter->value - parameter->grad * learningRate;
			}
		}
	}
}
