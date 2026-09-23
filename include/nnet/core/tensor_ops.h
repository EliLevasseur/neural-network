#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include "tensor.h"

// Rank contract for every operation in this file:
//
//   Leading dimensions are batch dimensions and are left untouched.
//   An operation acts on the last axis, or on the last two for anything
//   matrix-shaped.
//
// So a bias of shape {N} is added to the last axis of a tensor of any rank,
// transpose swaps the last two axes of any rank, and matmul multiplies the
// last two axes of each matrix in the stack. Nothing here broadcasts, and
// nothing here is limited to rank 2.

namespace nnet {
    Tensor fill(const Tensor::Shape& shape, double value);
    Tensor zeros(const Tensor::Shape& shape);

    Tensor sigmoid(const Tensor& tensor);
    Tensor sigmoidDerivitive(const Tensor& tensor);
    void addBias(Tensor& tensor, const Tensor& bias);
    Tensor addBiasGrad(const Tensor& upstreamGrad, const Tensor& bias);
    Tensor transpose(const Tensor& tensor);
    Tensor sumLeadingDimensions(const Tensor& tensor, std::size_t targetRank);
    Tensor matmulGradWeight(const Tensor& input, const Tensor& upstreamGrad);
    Tensor matmulGradInput(const Tensor& upstreamGrad, const Tensor& weight);
    double sum(const Tensor& tensor);
    std::vector<Tensor> createWeights(std::vector<size_t> layerSizes);
    Tensor createWeight(const size_t numInputs, const size_t numOutputs);
    Tensor createBias(const size_t layerCount);

    // BINARY PREDICTION MODEL FEATURES
    double binaryCrossEntropy(const Tensor& predictions, const Tensor& targets);
    Tensor binaryCrossEntropyGrad(const Tensor& predictions, const Tensor& targets);
}

#endif
