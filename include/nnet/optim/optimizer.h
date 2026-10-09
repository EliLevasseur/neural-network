#ifndef NNET_OPTIM_OPTIMIZER_H
#define NNET_OPTIM_OPTIMIZER_H

#include <vector>
#include "nnet/nn/parameter.h"

namespace nnet {

void sgdOptimizer(std::vector<Parameter*>& parameters, double learningRate);

class Optimizer {
	public:
		virtual ~Optimizer() = default;
		virtual void step() = 0;
};

class SGDOptimizer : public Optimizer {
	public:
		SGDOptimizer(std::vector<Parameter*> params, double learningRate);
		void step() override;
	private:
		std::vector<Parameter*> params_;
		double learningRate_;
};

class MomentumOptimizer : public Optimizer {
	public:
		MomentumOptimizer(std::vector<Parameter*> params, double learningRate, double beta = 0.9);
		void step() override;
	private:
		std::vector<Parameter*> params_;
		std::vector<Tensor> velocities_;
		double learningRate_;
		double beta_;
	};

// momentum's velocity plus an average of squared gradients
class AdamOptimizer : public Optimizer {
	public:
		AdamOptimizer(std::vector<Parameter*> params, double learningRate,
		              double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);
		void step() override;
	private:
		std::vector<Parameter*> params_;
		std::vector<Tensor> velocities_;
		std::vector<Tensor> squaredGradientAverages_;
		// per parameter so an unfrozen one starts its correction fresh
		std::vector<std::size_t> stepCounts_;
		double learningRate_;
		double beta1_;
		double beta2_;
		double epsilon_;
	};

}

#endif
