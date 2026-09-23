#include <iostream>
#include <vector>

#include "nnet/core/tensor.h"
#include "nnet/core/tensor_ops.h"
#include "nnet/nn/dense.h"
#include "dataframe.h"

// Eight inputs, two hidden layers, one output.
const std::vector<std::size_t> networkShape = {8, 6, 4, 1};

int main() {
    DataFrame data("data/complex_8d_test.csv", 8);
    const nnet::Tensor predictors = data.flatten(data.getPredictors());

    // Take the first row of the data as a single input.
    std::vector<double> firstRow;
    firstRow.reserve(networkShape[0]);
    for (std::size_t column = 0; column < networkShape[0]; column++) {
        firstRow.push_back(predictors.at({0, column}));
    }
    const nnet::Tensor input({1, networkShape[0]}, firstRow);

    // One Dense layer per neighbouring pair of sizes: 8->6, 6->4, 4->1.
    // Each layer creates its own weight and bias from those two numbers.
    std::vector<nnet::Dense> layers;
    layers.reserve(networkShape.size() - 1);
    for (std::size_t i = 0; i + 1 < networkShape.size(); i++) {
        layers.emplace_back(networkShape[i], networkShape[i + 1]);
    }

    // Walk the layers. Note that the activation is applied out here, by the
    // caller, and not inside Dense. That is what lets a different activation
    // be used later without touching the layer at all.
    nnet::Tensor activation = input;
    for (const nnet::Dense& layer : layers) {
        activation = nnet::sigmoid(layer.forward(activation));
    }

    std::cout << "input features : " << input.shape()[1] << '\n';
    std::cout << "layers         : " << layers.size() << '\n';
    std::cout << "output shape   : " << activation.shape()[0]
              << " x " << activation.shape()[1] << '\n';
    std::cout << "prediction     : " << activation.at({0, 0}) << '\n';

    return 0;
}
