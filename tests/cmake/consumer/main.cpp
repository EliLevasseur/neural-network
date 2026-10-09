#include <nnet/core/autograd/backward.h>
#include <nnet/nn/dense.h>
#include <nnet/optim/optimizer.h>

#include <cmath>
#include <iostream>
#include <random>

int main() {
    std::mt19937 generator(42);
    nnet::Dense layer(2, 1, generator);
    layer.weight.value = nnet::Tensor({2, 1}, {1.0, 2.0});
    const auto input = nnet::makeLeaf(nnet::Tensor({1, 2}, {3.0, 4.0}));
    const auto output = layer.forward(input);
    if (output->data.at({0, 0}) != 11.0) {
        return 1;
    }

    // One output entry: seed d(output)/d(output) = 1, then take one SGD step.
    nnet::backward(output);
    nnet::SGDOptimizer optimizer(layer.parameters(), 0.1);
    optimizer.step();

    nnet::NoGrad noGrad;
    const double updated = layer.forward(input)->data.at({0, 0});
    if (!(std::abs(updated - 8.4) < 1.0e-12)) {
        return 1;
    }
    std::cout << "Installed framework forward/backward/optimizer check passed\n";
}
