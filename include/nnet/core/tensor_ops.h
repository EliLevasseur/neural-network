#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include "tensor.h"

#include <random>

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

    Tensor relu(const Tensor& tensor);
    Tensor reluDerivitive(const Tensor& tensor);
    // Turns each row of scores into probabilities that add up to 1. Rows are
    // independent, so each image in a batch gets its own probabilities.
    Tensor softmax(const Tensor& scores);
    Tensor sigmoid(const Tensor& tensor);
    Tensor sigmoidDerivitive(const Tensor& tensor);
    void addBias(Tensor& tensor, const Tensor& bias);
    Tensor addBiasGrad(const Tensor& upstreamGrad, const Tensor& bias);
    Tensor transpose(const Tensor& tensor);
    Tensor sumLeadingDimensions(const Tensor& tensor, std::size_t targetRank);
    Tensor matmulGradWeight(const Tensor& input, const Tensor& upstreamGrad);
    Tensor matmulGradInput(const Tensor& upstreamGrad, const Tensor& weight);
    double sum(const Tensor& tensor);
    // The generator is borrowed, not copied, so layers built one after another
    // from the same generator continue its sequence instead of repeating it.
    Tensor createWeight(const size_t numInputs, const size_t numOutputs, std::mt19937& generator);
    Tensor createBias(const size_t layerCount);

    // BINARY PREDICTION MODEL FEATURES
    double binaryCrossEntropy(const Tensor& predictions, const Tensor& targets);
    Tensor binaryCrossEntropyGrad(const Tensor& predictions, const Tensor& targets);

    // MULTI-CLASS PREDICTION FEATURES
    // `scores` are the raw outputs of the last layer, one row per example.
    // `targets` has the same shape, and each of its rows must add up to 1;
    // a one-hot row (1 for the correct class, 0 elsewhere) does. The loss is
    // averaged over rows, like binaryCrossEntropy.
    double softmaxCrossEntropy(const Tensor& scores, const Tensor& targets);
    Tensor softmaxCrossEntropyGrad(const Tensor& scores, const Tensor& targets);
}

#endif
