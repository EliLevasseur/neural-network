#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include "tensor.h"

namespace nnet {
    Tensor sigmoid(const Tensor& tensor);
    double sum(const Tensor& tensor);
}

#endif
