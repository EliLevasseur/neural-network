#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include "tensor.h"
#include "nnet/nn/paramater.h"

namespace nnet {
    Tensor fill(const Tensor::Shape& shape, double value);
    Tensor zeros(const Tensor::Shape& shape);

    Tensor sigmoid(const Tensor& tensor);
    Tensor sigmoidDerivitive(const Tensor& tensor);
    void addBias(Tensor& tensor, const Tensor& bias);
    Tensor addBiasGrad(const Tensor& upstreamGrad, const Tensor& bias);
    Tensor transpose(const Tensor& tensor);
    Tensor matmulGradWeight(const Tensor& input, const Tensor& upstreamGrad);
    Tensor matmulGradInput(const Tensor& upstreamGrad, const Tensor& weight);
    double sum(const Tensor& tensor);
    std::vector<Paramater> createWeights(std::vector<size_t> layerSizes);
    Tensor createWeight(const size_t numInputs, const size_t numOutputs);

    // BINARY PREDICTION MODEL FEATURES
    double binaryCrossEntropy(const Tensor& predictions, const Tensor& targets);
    Tensor binaryCrossEntropyGrad(const Tensor& predictions, const Tensor& targets);
    Tensor feedForward(Tensor inputTensor, std::vector<Paramater>& weights);
}

#endif
