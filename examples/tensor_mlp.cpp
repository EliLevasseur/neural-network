#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "nnet/core/tensor.h"
#include "nnet/core/tensor_ops.h"
#include "nnet/nn/dense.h"
#include "dataframe.h"
#include "nnet/optim/optimizer.h"
#include "nnet/core/autograd/operations.h"
# include "nnet/core/autograd/backward.h"
#include "nnet/nn/module.h"
#include "nnet/nn/sequential.h"
#include "nnet/train/trainer.h"
#include "nnet/nn/checkpoint.h"

// Eight inputs, two hidden layers, one output.
const std::vector<std::size_t> networkShape = {8, 6, 4, 1};
const int epochs = 100;
const double learningRate = 0.5;
// One row per SGD step, as the legacy trainer does.
const std::size_t batchSize = 1;
const double trainSplit = 0.8;
// taken out of the training rows, not the test rows
const double validationFraction = 0.2;
const std::size_t randomSeed = 42;


int main() {
    DataFrame data("data/complex_8d_test.csv", 8);
    const splitContainer split = data.trainTestSplit(trainSplit, randomSeed);
    std::mt19937 validationGenerator(randomSeed);
    const nnet::TrainValidationSplit trainingData = nnet::splitOffValidation(
        data.flatten(split.XTrain), nnet::Tensor({split.yTrain.size(), 1}, split.yTrain),
        static_cast<std::size_t>(validationFraction * static_cast<double>(split.yTrain.size())),
        validationGenerator);
    const nnet::Tensor testPredictors = data.flatten(split.XTest);
    const nnet::Tensor testTargets({split.yTest.size(), 1}, split.yTest);

    std::mt19937 weightGenerator(randomSeed);
    std::vector<std::unique_ptr<nnet::Unary_Module>> components;
    for (std::size_t i = 0; i < networkShape.size() - 1; i++) {
        components.push_back(std::make_unique<nnet::Dense>(networkShape[i], networkShape[i + 1], weightGenerator));
        components.push_back(std::make_unique<nnet::Sigmoid>());
    }

    nnet::Sequential model(std::move(components));
    std::mt19937 shuffleGenerator(randomSeed);
    nnet::SGDOptimizer optimizer(model.parameters(), learningRate);
    for (std::size_t epoch = 0; epoch < epochs; epoch++) {
        const double trainingLoss = nnet::trainEpoch(model, trainingData.trainInputs, trainingData.trainTargets,
            batchSize, optimizer, nnet::binaryCrossEntropy, shuffleGenerator);
        const double validationLoss = nnet::evaluateLoss(model, trainingData.validationInputs,
            trainingData.validationTargets, nnet::binaryCrossEntropy);
        std::cout << "Epoch " << epoch + 1
                  << " | Training Loss: " << trainingLoss
                  << " | Validation Loss: " << validationLoss << '\n';
    }

    std::cout << "Test accuracy: " << nnet::accuracy(model, testPredictors, testTargets) << '\n'
              << "Test loss: "
              << nnet::evaluateLoss(model, testPredictors, testTargets, nnet::binaryCrossEntropy) << '\n';
    return 0;
}
