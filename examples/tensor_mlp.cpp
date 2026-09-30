#include <iostream>
#include <vector>

#include "nnet/core/tensor.h"
#include "nnet/core/tensor_ops.h"
#include "nnet/nn/dense.h"
#include "dataframe.h"
#include "optim/sgd.h"

// Eight inputs, two hidden layers, one output.
const std::vector<std::size_t> networkShape = {8, 6, 4, 1};
const int epochs = 1000;
const double learningRate = 0.5;

struct ForwardTrace {
    std::vector<nnet::Tensor> activations;     // the input, then one per layer
    std::vector<nnet::Tensor> preActivations;  // one per layer
};


ForwardTrace forwardPass(const std::vector<nnet::Dense>& layers,
                         const nnet::Tensor& input) {
    ForwardTrace trace;
    trace.activations.reserve(layers.size() + 1);
    trace.preActivations.reserve(layers.size());

    trace.activations.push_back(input);

    for (const nnet::Dense& layer : layers) {
        trace.preActivations.push_back(layer.forward(trace.activations.back()));
        trace.activations.push_back(nnet::sigmoid(trace.preActivations.back()));
    }
    return trace;
}

void backwardPass(std::vector<nnet::Dense>& layers,
                  const ForwardTrace& trace,
                  const nnet::Tensor& target) {

    nnet::Tensor gradient =
        nnet::binaryCrossEntropyGrad(trace.activations.back(), target);


    for (std::size_t index = layers.size(); index > 0;) {
        --index;

        // The activation's derivative belongs out here, not inside Dense,
        // exactly as applying the activation does on the way forward.
        gradient.inplaceMultiplication(
            nnet::sigmoidDerivitive(trace.preActivations[index]));

        // The layer fills in its own two gradients and hands back the one
        // belonging to its input, which is the layer before it.
        gradient = layers[index].backward(trace.activations[index], gradient);
    }
}

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

void updateNetwork(std::vector<nnet::Dense>& network, double learningRate) {
    for (auto& layer : network) {
        auto parameters = layer.parameters();
        nnet::sgdOptimizer(parameters, learningRate);
    }
}



int main() {
    DataFrame data("data/complex_8d_test.csv", 8);
    const nnet::Tensor predictors = data.flatten(data.getPredictors());
    const nnet::Tensor targets({data.getTargets().size(), 1}, data.getTargets());

    std::vector<nnet::Dense> network;
    network.reserve(networkShape.size() - 1);

    for (std::size_t i = 0; i < networkShape.size() - 1; i++) {
        network.emplace_back(networkShape[i], networkShape[i + 1]);
    }

    for (std::size_t epoch = 0; epoch < epochs; epoch++) {
        for (std::size_t row = 0; row < predictors.shape()[0]; ++row) {
            const nnet::Tensor input = makeInputRow(predictors, row);
            const nnet::Tensor target = makeTargetRow(targets, row);
            const ForwardTrace trace = forwardPass(network, input);

            backwardPass(network, trace, target);
            updateNetwork(network, learningRate);
        }

        const ForwardTrace evaluation = forwardPass(network, predictors);
        const double loss =
            nnet::binaryCrossEntropy(evaluation.activations.back(), targets);
        std::cout << "Epoch " << epoch + 1
                  << " | Loss: " << loss << '\n';
    }

    return 0;

}
