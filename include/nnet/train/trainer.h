#ifndef NNET_TRAIN_TRAINER_H
#define NNET_TRAIN_TRAINER_H

#include "nnet/core/autograd/value.h"
#include "nnet/nn/module.h"
#include "nnet/optim/optimizer.h"

#include <cstddef>
#include <random>
#include <vector>

// inputs are [rows, features], targets are [rows, columns]

namespace nnet {

    // binaryCrossEntropy or softmaxCrossEntropy
    using LossFunction = Value (*)(const Value& prediction, const Value& target);

    // copies the listed rows, in that order
    Tensor takeRows(const Tensor& data, const std::vector<std::size_t>& rowIndices);

    struct TrainValidationSplit {
        Tensor trainInputs;
        Tensor trainTargets;
        Tensor validationInputs;
        Tensor validationTargets;
    };

    // same idea as DataFrame::trainTestSplit: shuffle the rows, cut off validationRows of them
    TrainValidationSplit splitOffValidation(const Tensor& inputs, const Tensor& targets,
                                            std::size_t validationRows, std::mt19937& generator);

    // one shuffled pass in batches, one optimizer step per batch; returns the average loss
    double trainEpoch(Unary_Module& model, const Tensor& inputs, const Tensor& targets,
                      std::size_t batchSize, Optimizer& optimizer,
                      LossFunction loss, std::mt19937& generator);

    double evaluateLoss(const Unary_Module& model, const Tensor& inputs, const Tensor& targets,
                        LossFunction loss);

    // one column: >= 0.5 counts as yes. several columns: highest score is the guess
    double accuracy(const Unary_Module& model, const Tensor& inputs, const Tensor& targets);

}

#endif
