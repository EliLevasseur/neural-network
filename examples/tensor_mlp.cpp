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
const int epochs = 1000;
const double learningRate = 0.5;


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
    const nnet::Tensor predictors = data.flatten(data.getPredictors());
    const nnet::Tensor targets({data.getTargets().size(), 1}, data.getTargets());

    std::vector<std::unique_ptr<nnet::Unary_Module>> components;
    for (std::size_t i = 0; i < networkShape.size() - 1; i++) {
        components.push_back(std::make_unique<nnet::Dense>(networkShape[i], networkShape[i + 1]));
        components.push_back(std::make_unique<nnet::Sigmoid>());
    }

    nnet::Sequential model(std::move(components));
    std::vector<nnet::Parameter*> parameters = model.parameters();

    for (std::size_t epoch = 0; epoch < epochs; epoch++) {
        for (std::size_t row = 0; row < predictors.shape()[0]; ++row) {
            const nnet::Tensor input = makeInputRow(predictors, row);
            const nnet::Tensor target = makeTargetRow(targets, row);

            model.zeroGrad();
            nnet::Value prediction = model.forward(nnet::makeLeaf(input));
            const nnet::Value loss = nnet::binaryCrossEntropy(prediction, nnet::makeLeaf(target));
            nnet::backward(loss);
            sgdOptimizer(parameters, learningRate);          
        }
        
        nnet::NoGrad noGrad;
        nnet::Value evaluation = model.forward(nnet::makeLeaf(predictors));
        double loss = nnet::binaryCrossEntropy(evaluation->data, targets);     
           
        std::cout << "Epoch " << epoch + 1
                  << " | Loss: " << loss << '\n';
    }

    return 0;

}
