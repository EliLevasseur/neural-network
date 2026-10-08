#ifndef NNET_TRAIN_TRAINER_H
#define NNET_TRAIN_TRAINER_H

#include "nnet/core/autograd/value.h"
#include "nnet/nn/module.h"

#include <cstddef>
#include <random>
#include <vector>

// The training loop every example shares. These are plain functions rather
// than a class because there is nothing for a trainer to own: the model owns
// its parameters, autograd builds and releases a graph per step, and the
// caller owns the data and the random generator.
//
// Data comes in as two rank-2 Tensors with one row per example: `inputs` is
// [rows, features] and `targets` is [rows, columns]. Any loader that produces
// those (DataFrame for CSVs, an image reader, ...) works with these functions.

namespace nnet {

    // Any loss shaped like binaryCrossEntropy and softmaxCrossEntropy: it takes
    // the model's output and the targets, and returns a {1}-shaped Value.
    using LossFunction = Value (*)(const Value& prediction, const Value& target);

    // Copies the listed rows into a new Tensor, in the order given. The result
    // keeps every dimension except the first, which becomes rowIndices.size().
    Tensor takeRows(const Tensor& data, const std::vector<std::size_t>& rowIndices);

    // One pass over every row, in an order shuffled by `generator`, cut into
    // batches of `batchSize` rows (the last one may be shorter). Each batch
    // runs zeroGrad, forward, loss, backward, and one SGD step. Returns the
    // loss averaged over every row, as measured while training.
    double trainEpoch(Unary_Module& model, const Tensor& inputs, const Tensor& targets,
                      std::size_t batchSize, double learningRate,
                      LossFunction loss, std::mt19937& generator);

    // The loss over all rows at once, without recording a graph.
    double evaluateLoss(const Unary_Module& model, const Tensor& inputs, const Tensor& targets,
                        LossFunction loss);

    // The fraction of rows the model gets right, without recording a graph.
    // With one target column, the output is read as a probability and counts
    // as "yes" at 0.5 or above. With several columns, the column with the
    // highest score is the guess, which matches a one-hot target's 1.
    double accuracy(const Unary_Module& model, const Tensor& inputs, const Tensor& targets);

}

#endif
