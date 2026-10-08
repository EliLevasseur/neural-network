#include "test_utils.h"
#include "nnet/nn/dense.h"
#include "optim/sgd.h"
#include "nnet/core/autograd/backward.h"

#include <random>

// Standalone integration test; links the real optimizer, not an inline copy.
// Build from the repository root:
// g++ -std=c++17 -Wall -Wextra -Wpedantic -Iinclude -Itests tests/optimizer_test.cpp src/core/tensor.cpp src/core/tensor_ops.cpp src/optim/sgd.cpp src/autograd/operations.cpp src/autograd/backward.cpp -o build/optimizer_tests
// Run: ./build/optimizer_tests

namespace {
    // Builds the layer's graph and runs backward from a known incoming
    // gradient, returning the gradient that reached the input.
    nnet::Tensor backwardThrough(const nnet::Dense& layer, const nnet::Tensor& input,
                                 const nnet::Tensor& upstream) {
        nnet::Value inputLeaf = nnet::makeLeaf(input);
        nnet::backward(layer.forward(inputLeaf), upstream);
        return inputLeaf->grad;
    }
}

int main() {
    TestRunner tests;
    tests.section("FROZEN PARAMETERS / DENSE / SGD");

    std::mt19937 generator(1);
    nnet::Dense layer(1, 1, generator);
    layer.weight.value = nnet::Tensor({1, 1}, {0.5});
    layer.bias.value = nnet::Tensor({1}, {0.25});
    const nnet::Tensor input({1, 1}, {2.0});
    const nnet::Tensor upstream({1, 1}, {3.0});
    const double tolerance = 1.0e-12;
    const double learningRate = 0.1;

    tests.expectTrue(layer.weight.requiresGrad && layer.bias.requiresGrad,
        "parameters are trainable by default");

    // Simulate freezing a weight that still has a gradient from earlier work.
    layer.zeroGrad();
    layer.weight.requiresGrad = false;
    layer.weight.grad.at({0, 0}) = 9.0;

    tests.expectNear(layer.forward(nnet::makeLeaf(input))->data.at({0, 0}), 1.25, tolerance,
        "forward still uses the frozen weight");

    const nnet::Tensor inputGradient = backwardThrough(layer, input, upstream);
    tests.expectNear(layer.weight.grad.at({0, 0}), 9.0, tolerance,
        "backward adds no contribution to the frozen weight");
    tests.expectNear(layer.bias.grad.at({0}), 3.0, tolerance,
        "backward still calculates the trainable bias gradient");
    tests.expectNear(inputGradient.at({0, 0}), 1.5, tolerance,
        "backward passes the input gradient through the frozen weight");

    auto parameters = layer.parameters();
    tests.expectTrue(parameters == std::vector<nnet::Parameter*>{&layer.weight, &layer.bias},
        "discovery includes frozen as well as trainable parameters");
    nnet::sgdOptimizer(parameters, learningRate);

    tests.expectNear(layer.weight.value.at({0, 0}), 0.5, tolerance,
        "SGD preserves a frozen weight despite its stale nonzero gradient");
    tests.expectNear(layer.bias.value.at({0}), -0.05, tolerance,
        "SGD updates the trainable bias using its computed gradient");

    // Reverse the flags to check that the two parameters are independent.
    layer.weight.requiresGrad = true;
    layer.bias.requiresGrad = false;
    layer.bias.value.at({0}) = 0.25;
    layer.zeroGrad();
    tests.expectTrue(layer.weight.grad.at({0, 0}) == 0.0 &&
        layer.bias.grad.at({0}) == 0.0,
        "zeroGrad clears both frozen and trainable gradients");
    layer.bias.grad.at({0}) = 7.0;

    const nnet::Tensor unfrozenInputGradient = backwardThrough(layer, input, upstream);
    tests.expectNear(layer.weight.grad.at({0, 0}), 6.0, tolerance,
        "unfreezing the weight restores its gradient contribution");
    tests.expectNear(layer.bias.grad.at({0}), 7.0, tolerance,
        "backward adds no contribution to the frozen bias");
    tests.expectNear(unfrozenInputGradient.at({0, 0}), 1.5, tolerance,
        "freezing the bias leaves the input gradient intact");

    // Reuse the previously discovered pointers: changing a flag does not
    // change Parameter identity, and SGD must inspect the current flags.
    nnet::sgdOptimizer(parameters, learningRate);
    tests.expectNear(layer.weight.value.at({0, 0}), -0.1, tolerance,
        "SGD updates the unfrozen weight using its computed gradient");
    tests.expectNear(layer.bias.value.at({0}), 0.25, tolerance,
        "SGD preserves a frozen bias despite its stale nonzero gradient");

    // A frozen parameter is still subject to its shape contract. Validate
    // every entry before updating the first one, including the frozen entries.
    layer.weight.grad.at({0, 0}) = 2.0;
    const double beforeInvalidStep = layer.weight.value.at({0, 0});
    layer.bias.value = nnet::Tensor({2}, {0.25, 0.25});
    layer.bias.grad = nnet::Tensor({2}, {7.0, 7.0});
    bool rejectedShape = false;
    try { nnet::sgdOptimizer(parameters, learningRate); }
    catch (const std::invalid_argument&) { rejectedShape = true; }
    tests.expectTrue(rejectedShape, "SGD rejects resized frozen value/grad tensors");
    tests.expectNear(layer.weight.value.at({0, 0}), beforeInvalidStep, 0.0,
        "SGD validates the entire parameter list before updating any entry");

    std::vector<nnet::Parameter*> invalidPointers = {&layer.weight, nullptr};
    bool rejectedNull = false;
    try { nnet::sgdOptimizer(invalidPointers, learningRate); }
    catch (const std::invalid_argument&) { rejectedNull = true; }
    tests.expectTrue(rejectedNull, "SGD rejects null parameter pointers");
    tests.expectNear(layer.weight.value.at({0, 0}), beforeInvalidStep, 0.0,
        "null parameter rejection does not partially update the model");

    return tests.finish();
}
