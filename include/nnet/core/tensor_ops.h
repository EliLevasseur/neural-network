#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include "tensor.h"

namespace nnet {
    Tensor sigmoid(const Tensor& tensor);
    std::vector<Tensor> createWeights(const Tensor& tensor, const Tensor::Shape& networkShape);
    Tensor createBiases(const Tensor& tensor);

    Tensor forwardPass(const Tensor& input, const Tensor& weights, const Tensor& biases);
    

    double sum(const Tensor& tensor);
}

#endif
