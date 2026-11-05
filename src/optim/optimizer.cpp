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
		// Updated in place: no temporary tensors.
		for (auto& parameter : parameters) {
			if (parameter->requiresGrad) {
				double* value = parameter->value.data();
				const double* grad = parameter->grad.data();
				for (std::size_t i = 0; i < parameter->value.numel(); i++) {
					value[i] -= grad[i] * learningRate;
				}
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
			if (!params_[i]->requiresGrad) {
				continue;
			}
			double* value = params_[i]->value.data();
			const double* grad = params_[i]->grad.data();
			double* velocity = velocities_[i].data();
			for (std::size_t k = 0; k < params_[i]->value.numel(); k++) {
				velocity[k] = velocity[k] * beta_ + grad[k] * (1 - beta_);
				value[k] -= velocity[k] * learningRate_;
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

			// One loop updates both averages and the value, in place.
			double* value = params_[i]->value.data();
			const double* grad = params_[i]->grad.data();
			double* velocity = velocities_[i].data();
			double* squaredAverage = squaredGradientAverages_[i].data();
			for (std::size_t k = 0; k < params_[i]->value.numel(); k++) {
				velocity[k] = velocity[k] * beta1_ + grad[k] * (1 - beta1_);
				squaredAverage[k] = squaredAverage[k] * beta2_ + (grad[k] * grad[k]) * (1 - beta2_);
				const double correctedVelocity = velocity[k] / velocityCorrection;
				const double correctedSquared = squaredAverage[k] / squaredCorrection;
				value[k] -= learningRate_ * correctedVelocity / (std::sqrt(correctedSquared) + epsilon_);
			}
		}
	}

}
