#include "nnet/optim/optimizer.h"

#include <cmath>
#include <stdexcept>
#include <utility>

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

	SGDOptimizer::SGDOptimizer(std::vector<Parameter*> params, double learningRate) : params_{std::move(params)}, learningRate_{learningRate} {}

	void SGDOptimizer::step() {
		sgdOptimizer(params_, learningRate_);
	}

	MomentumOptimizer::MomentumOptimizer(std::vector<Parameter*> params, double learningRate, double beta) : params_{std::move(params)}, learningRate_{learningRate}, beta_{beta} {
		for (auto* param : params_) {
			if (!param)
				throw std::invalid_argument("Received a null parameter");
			velocities_.push_back(zeros(param->value.shape()));
		}
	}

	void MomentumOptimizer::step() {
		// check everything first so a bad parameter can't leave a half-finished step
		for (const auto* param : params_) {
			param->validateShape();
		}
		for (std::size_t i = 0; i < params_.size(); i++) {
			if (params_[i]->requiresGrad) {
				velocities_[i] = (velocities_[i]*beta_) + params_[i]->grad * (1 - beta_);
				params_[i]->value = params_[i]->value - (velocities_[i] * learningRate_);
			}
		}
	}

	AdamOptimizer::AdamOptimizer(std::vector<Parameter*> params, double learningRate, double beta1, double beta2, double epsilon) : params_{std::move(params)}, learningRate_{learningRate}, beta1_{beta1}, beta2_{beta2}, epsilon_{epsilon} {
		for (auto* param : params_) {
			if (!param)
				throw std::invalid_argument("Adam received a null Parameter");
			velocities_.push_back(zeros(param->value.shape()));
			squaredGradientAverages_.push_back(zeros(param->value.shape()));
			stepCounts_.push_back(0);
		}
	}

	void AdamOptimizer::step() {
		for (const auto* param : params_) {
			param->validateShape();
		}
		for (std::size_t i = 0; i < params_.size(); i++) {
			if (!params_[i]->requiresGrad) {
				continue;
			}
			stepCounts_[i]++;
			const double velocityCorrection = 1.0 - std::pow(beta1_, static_cast<double>(stepCounts_[i]));
			const double squaredCorrection = 1.0 - std::pow(beta2_, static_cast<double>(stepCounts_[i]));

			Tensor gradientSquared = params_[i]->grad;
			gradientSquared.inplaceMultiplication(params_[i]->grad);
			velocities_[i] = (velocities_[i] * beta1_) + params_[i]->grad * (1 - beta1_);
			squaredGradientAverages_[i] = (squaredGradientAverages_[i] * beta2_) + gradientSquared * (1 - beta2_);

			// no elementwise sqrt or divide on Tensor yet, so do it per number
			const std::vector<double>& values = params_[i]->value.getData();
			const std::vector<double>& velocity = velocities_[i].getData();
			const std::vector<double>& squaredAverage = squaredGradientAverages_[i].getData();
			std::vector<double> newValues(values.size());
			for (std::size_t k = 0; k < values.size(); k++) {
				const double correctedVelocity = velocity[k] / velocityCorrection;
				const double correctedSquared = squaredAverage[k] / squaredCorrection;
				newValues[k] = values[k] - learningRate_ * correctedVelocity / (std::sqrt(correctedSquared) + epsilon_);
			}
			params_[i]->value = Tensor(params_[i]->value.shape(), std::move(newValues));
		}
	}

}
