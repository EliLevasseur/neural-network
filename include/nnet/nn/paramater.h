#ifndef PARAMATER_H
#define PARAMATER_H

#include "nnet/core/tensor.h"
#include <utility>

namespace nnet {

    struct Paramater {
        Tensor value;
        Tensor grad;
    
        explicit Paramater(Tensor initialValue)
        : value(std::move(initialValue)),
          grad(value.shape(), std::vector<double>(value.numel(), 0.0)) {}
          
    };
}

#endif