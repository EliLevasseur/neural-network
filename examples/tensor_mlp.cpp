#include <iostream>
#include <memory>
#include <vector>

#include "nnet/core/tensor.h"
#include "nnet/core/tensor_ops.h"
#include "nnet/nn/dense.h"
#include "dataframe.h"
#include "optim/sgd.h"
#include "nnet/core/autograd/operations.h"
# include "nnet/core/autograd/backward.h"
#include "nnet/nn/module.h"
#include "nnet/nn/sequential.h"

// Eight inputs, two hidden layers, one output.
const std::vector<std::size_t> networkShape = {8, 6, 4, 1};
const int epochs = 100;
const double learningRate = 0.5;
const double trainSplit = 0.8;
const std::size_t randomSeed = 42;


nnet::Tensor makeInputRow(const nnet::Tensor& predictors, std::size_t row) {
    const std::size_t width = predictors.shape()[1];
    std::vector<double> values;
    values.reserve(width);
    for (std::size_t column = 0; column < width; ++column) {
        values.push_back(predictors.at({row, column}));
    }
    return nnet::Tensor({1, width}, values);
}

nnet::Tensor makeTargetRow(const nnet::Tensor& targets, std::size_t row) {
    return nnet::Tensor({1, 1}, {targets.at({row, 0})});
}

int main() {
    DataFrame data("data/complex_8d_test.csv", 8);
    const splitContainer split = data.trainTestSplit(trainSplit, randomSeed);
    const nnet::Tensor trainPredictors = data.flatten(split.XTrain);
    const nnet::Tensor trainTargets({split.yTrain.size(), 1}, split.yTrain);
    const nnet::Tensor testPredictors = data.flatten(split.XTest);
    const nnet::Tensor testTargets({split.yTest.size(), 1}, split.yTest);

    std::vector<std::unique_ptr<nnet::Unary_Module>> components;
    for (std::size_t i = 0; i < networkShape.size() - 1; i++) {
        components.push_back(std::make_unique<nnet::Dense>(networkShape[i], networkShape[i + 1]));
        components.push_back(std::make_unique<nnet::Sigmoid>());
    }

    nnet::Sequential model(std::move(components));
    std::vector<nnet::Parameter*> parameters = model.parameters();

    for (std::size_t epoch = 0; epoch < epochs; epoch++) {
        for (std::size_t row = 0; row < trainPredictors.shape()[0]; ++row) {
            const nnet::Tensor input = makeInputRow(trainPredictors, row);
            const nnet::Tensor target = makeTargetRow(trainTargets, row);

            model.zeroGrad();
            nnet::Value prediction = model.forward(nnet::makeLeaf(input));
            const nnet::Value loss = nnet::binaryCrossEntropy(prediction, nnet::makeLeaf(target));
            nnet::backward(loss);
            sgdOptimizer(parameters, learningRate);
        }

        nnet::NoGrad noGrad;
        nnet::Value evaluation = model.forward(nnet::makeLeaf(trainPredictors));
        const double loss = nnet::binaryCrossEntropy(evaluation->data, trainTargets);

        std::cout << "Epoch " << epoch + 1
                  << " | Training Loss: " << loss << '\n';
    }

    nnet::NoGrad noGrad;
    const nnet::Value testPredictions = model.forward(nnet::makeLeaf(testPredictors));
    const double testLoss = nnet::binaryCrossEntropy(testPredictions->data, testTargets);

    std::size_t correctPredictions = 0;
    const std::vector<double>& predictions = testPredictions->data.getData();
    const std::vector<double>& labels = testTargets.getData();
    for (std::size_t row = 0; row < predictions.size(); ++row) {
        if ((predictions[row] >= 0.5) == (labels[row] >= 0.5)) {
            ++correctPredictions;
        }
    }

    const double testAccuracy =
        static_cast<double>(correctPredictions) / static_cast<double>(predictions.size());
    std::cout << "Test accuracy: " << testAccuracy << '\n'
              << "Test loss: " << testLoss << '\n';

    return 0;
}
