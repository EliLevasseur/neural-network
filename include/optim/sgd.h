#ifndef NNET_OPTIM_SGD_H
#define NNET_OPTIM_SGD_H

#include <vector>
#include "nnet/nn/parameter.h"

namespace nnet {

void sgdOptimizer(std::vector<Parameter*>& parameters, double learningRate);

}

#endif
