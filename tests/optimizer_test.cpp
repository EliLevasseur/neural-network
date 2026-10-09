#include "test_utils.h"
#include "nnet/nn/dense.h"
#include "nnet/optim/optimizer.h"
#include "nnet/core/autograd/backward.h"

#include <random>
#include <stdexcept>
#include <string>

// Standalone integration test; links the real optimizer, not an inline copy.
// Build from the repository root:
// g++ -std=c++17 -Wall -Wextra -Wpedantic -Iinclude -Itests tests/optimizer_test.cpp src/core/tensor.cpp src/core/tensor_ops.cpp src/optim/optimizer.cpp src/autograd/operations.cpp src/autograd/backward.cpp -o build/optimizer_tests
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

    tests.section("MOMENTUM");

    // ---- known numbers over three steps ------------------------------------
    // Gradient 1 every step, learning rate 0.1, beta 0.9, starting at 0:
    //   step 1: velocity = 0.9 * 0    + 0.1 * 1 = 0.1     weight = 0      - 0.1 * 0.1   = -0.01
    //   step 2: velocity = 0.9 * 0.1  + 0.1 * 1 = 0.19    weight = -0.01  - 0.1 * 0.19  = -0.029
    //   step 3: velocity = 0.9 * 0.19 + 0.1 * 1 = 0.271   weight = -0.029 - 0.1 * 0.271 = -0.0561
    // Momentum's whole point is remembering the velocity between steps. A
    // version that forgot it would give -0.01, -0.02, -0.03 instead.
    {
        nnet::Parameter weight(nnet::Tensor({1, 1}, {0.0}));
        nnet::MomentumOptimizer optimizer({&weight}, 0.1);
        const double expected[3] = {-0.01, -0.029, -0.0561};
        for (int step = 0; step < 3; ++step) {
            weight.grad = nnet::Tensor({1, 1}, {1.0});
            optimizer.step();
            tests.expectNear(weight.value.at({0, 0}), expected[step], tolerance,
                "momentum matches the hand-worked weight after step " + std::to_string(step + 1));
        }
    }

    // ---- every number in a Parameter has its own velocity ------------------
    // Gradients 1 and -2 for the two numbers. After two steps the velocities
    // are 0.19 and -0.38, so the weights are -0.029 and 0.058.
    {
        nnet::Parameter weight(nnet::Tensor({2}, {0.0, 0.0}));
        nnet::MomentumOptimizer optimizer({&weight}, 0.1);
        for (int step = 0; step < 2; ++step) {
            weight.grad = nnet::Tensor({2}, {1.0, -2.0});
            optimizer.step();
        }
        tests.expectNear(weight.value.at({0}), -0.029, tolerance,
            "momentum gives the first number its own velocity");
        tests.expectNear(weight.value.at({1}), 0.058, tolerance,
            "momentum gives the second number its own velocity");
    }

    // ---- every Parameter has its own velocity ------------------------------
    // Only the first Parameter has a gradient. The second must not move, which
    // would fail if velocities were shared or matched to the wrong Parameter.
    {
        nnet::Parameter moving(nnet::Tensor({1, 1}, {0.0}));
        nnet::Parameter still(nnet::Tensor({1, 1}, {0.0}));
        nnet::MomentumOptimizer optimizer({&moving, &still}, 0.1);
        for (int step = 0; step < 2; ++step) {
            moving.grad = nnet::Tensor({1, 1}, {1.0});
            still.grad = nnet::Tensor({1, 1}, {0.0});
            optimizer.step();
        }
        tests.expectNear(moving.value.at({0, 0}), -0.029, tolerance,
            "momentum moves the Parameter that has a gradient");
        tests.expectNear(still.value.at({0, 0}), 0.0, 0.0,
            "momentum leaves a Parameter with zero gradient where it is");
    }

    // ---- frozen Parameters -------------------------------------------------
    // A frozen Parameter keeps its value even with a nonzero gradient, and
    // builds up no velocity while frozen: once unfrozen, its first step is
    // the same as a fresh start (5 - 0.1 * 0.1 = 4.99).
    {
        nnet::Parameter trainable(nnet::Tensor({1, 1}, {0.0}));
        nnet::Parameter frozen(nnet::Tensor({1, 1}, {5.0}));
        frozen.requiresGrad = false;
        nnet::MomentumOptimizer optimizer({&trainable, &frozen}, 0.1);
        for (int step = 0; step < 2; ++step) {
            trainable.grad = nnet::Tensor({1, 1}, {1.0});
            frozen.grad = nnet::Tensor({1, 1}, {1.0});
            optimizer.step();
        }
        tests.expectNear(frozen.value.at({0, 0}), 5.0, 0.0,
            "momentum preserves a frozen Parameter despite its gradient");
        tests.expectNear(trainable.value.at({0, 0}), -0.029, tolerance,
            "momentum still updates the trainable Parameter next to a frozen one");

        frozen.requiresGrad = true;
        frozen.grad = nnet::Tensor({1, 1}, {1.0});
        optimizer.step();
        tests.expectNear(frozen.value.at({0, 0}), 4.99, tolerance,
            "a Parameter builds no velocity while frozen");
    }

    // ---- invalid shapes: rejected, and nothing moves -----------------------
    // The second Parameter's gradient has the wrong shape. The step must be
    // refused before the first Parameter is touched.
    {
        nnet::Parameter first(nnet::Tensor({1, 1}, {0.0}));
        nnet::Parameter second(nnet::Tensor({1, 1}, {0.0}));
        nnet::MomentumOptimizer optimizer({&first, &second}, 0.1);
        first.grad = nnet::Tensor({1, 1}, {1.0});
        second.grad = nnet::Tensor({2, 1}, {1.0, 1.0});
        bool rejected = false;
        try { optimizer.step(); }
        catch (const std::invalid_argument&) { rejected = true; }
        tests.expectTrue(rejected, "momentum rejects a gradient whose shape changed");
        tests.expectNear(first.value.at({0, 0}), 0.0, 0.0,
            "momentum checks every Parameter before updating any");
    }

    // ---- null Parameters: rejected when the optimizer is created -----------
    {
        bool rejected = false;
        try { nnet::MomentumOptimizer optimizer({nullptr}, 0.1); }
        catch (const std::invalid_argument&) { rejected = true; }
        tests.expectTrue(rejected, "momentum rejects a null Parameter");
    }

    tests.section("ADAM");

    // ---- known numbers over two steps --------------------------------------
    // Gradients 1 then 3, learning rate 0.1, beta1 0.9, beta2 0.999:
    //   step 1: velocity 0.1, squared average 0.001. Corrected by 0.1 and
    //           0.001 they become 1 and 1, so the step is 0.1 * 1 / 1 = 0.1.
    //   step 2: velocity 0.9 * 0.1 + 0.1 * 3 = 0.39, corrected 0.39 / 0.19 = 2.0526
    //           squared  0.999 * 0.001 + 0.001 * 9 = 0.009999, corrected / 0.001999 = 5.0020
    //           step 0.1 * 2.0526 / sqrt(5.0020) = 0.0918, so the weight is -0.1918
    // The epsilon of 1e-8 nudges the last digits: -0.099999999 and -0.191778110.
    {
        nnet::Parameter weight(nnet::Tensor({1, 1}, {0.0}));
        nnet::AdamOptimizer optimizer({&weight}, 0.1);
        weight.grad = nnet::Tensor({1, 1}, {1.0});
        optimizer.step();
        tests.expectNear(weight.value.at({0, 0}), -0.099999999, 1.0e-9,
            "Adam's corrected first step is a full learning-rate step");
        weight.grad = nnet::Tensor({1, 1}, {3.0});
        optimizer.step();
        tests.expectNear(weight.value.at({0, 0}), -0.191778110488, 1.0e-9,
            "Adam matches the hand-worked weight after step 2");
    }

    // ---- the size of a gradient does not set the size of the step ----------
    // Plain SGD at learning rate 0.1 would move these two numbers by 10 and by
    // 0.001 each step. Adam moves both by about 0.1 each step.
    {
        nnet::Parameter weight(nnet::Tensor({2}, {0.0, 0.0}));
        nnet::AdamOptimizer optimizer({&weight}, 0.1);
        for (int step = 0; step < 3; ++step) {
            weight.grad = nnet::Tensor({2}, {100.0, 0.01});
            optimizer.step();
        }
        tests.expectNear(weight.value.at({0}), -0.3, 1.0e-6,
            "Adam moves a weight with gradient 100 by the learning rate each step");
        tests.expectNear(weight.value.at({1}), -0.3, 1.0e-6,
            "Adam moves a weight with gradient 0.01 by the learning rate each step");
    }

    // ---- a gradient that is always zero leaves the weight alone ------------
    // Both averages stay 0. Without epsilon the update would be 0 / 0 = NaN.
    {
        nnet::Parameter weight(nnet::Tensor({1, 1}, {0.25}));
        nnet::AdamOptimizer optimizer({&weight}, 0.1);
        for (int step = 0; step < 2; ++step) {
            weight.grad = nnet::Tensor({1, 1}, {0.0});
            optimizer.step();
        }
        tests.expectNear(weight.value.at({0, 0}), 0.25, 0.0,
            "Adam leaves a weight whose gradient is always zero unchanged, not NaN");
    }

    // ---- frozen Parameters -------------------------------------------------
    // A frozen Parameter keeps its value, and takes no steps of its own while
    // frozen. Once unfrozen, its first step is a fresh start: 5 - 0.1 = 4.9.
    // A single shared step count would give 4.936 here instead.
    {
        nnet::Parameter trainable(nnet::Tensor({1, 1}, {0.0}));
        nnet::Parameter frozen(nnet::Tensor({1, 1}, {5.0}));
        frozen.requiresGrad = false;
        nnet::AdamOptimizer optimizer({&trainable, &frozen}, 0.1);
        for (int step = 0; step < 2; ++step) {
            trainable.grad = nnet::Tensor({1, 1}, {1.0});
            frozen.grad = nnet::Tensor({1, 1}, {1.0});
            optimizer.step();
        }
        tests.expectNear(frozen.value.at({0, 0}), 5.0, 0.0,
            "Adam preserves a frozen Parameter despite its gradient");
        tests.expectNear(trainable.value.at({0, 0}), -0.2, 1.0e-7,
            "Adam still updates the trainable Parameter next to a frozen one");

        frozen.requiresGrad = true;
        trainable.grad = nnet::Tensor({1, 1}, {1.0});
        frozen.grad = nnet::Tensor({1, 1}, {1.0});
        optimizer.step();
        tests.expectNear(frozen.value.at({0, 0}), 4.9, 1.0e-7,
            "an unfrozen Parameter's first Adam step is a fresh start");
    }

    // ---- invalid shapes: rejected, and nothing moves -----------------------
    {
        nnet::Parameter first(nnet::Tensor({1, 1}, {0.0}));
        nnet::Parameter second(nnet::Tensor({1, 1}, {0.0}));
        nnet::AdamOptimizer optimizer({&first, &second}, 0.1);
        first.grad = nnet::Tensor({1, 1}, {1.0});
        second.grad = nnet::Tensor({2, 1}, {1.0, 1.0});
        bool rejected = false;
        try { optimizer.step(); }
        catch (const std::invalid_argument&) { rejected = true; }
        tests.expectTrue(rejected, "Adam rejects a gradient whose shape changed");
        tests.expectNear(first.value.at({0, 0}), 0.0, 0.0,
            "Adam checks every Parameter before updating any");
    }

    // ---- null Parameters: rejected when the optimizer is created -----------
    {
        bool rejected = false;
        try { nnet::AdamOptimizer optimizer({nullptr}, 0.1); }
        catch (const std::invalid_argument&) { rejected = true; }
        tests.expectTrue(rejected, "Adam rejects a null Parameter");
    }

    tests.section("OPTIMIZER INTERFACE");

    // 0.5 - 0.1 * 2 = 0.3
    {
        nnet::Parameter weight(nnet::Tensor({1, 1}, {0.5}));
        nnet::SGDOptimizer optimizer({&weight}, 0.1);
        weight.grad = nnet::Tensor({1, 1}, {2.0});
        optimizer.step();
        tests.expectNear(weight.value.at({0, 0}), 0.3, tolerance, "SGDOptimizer takes a plain SGD step");
    }

    // same calls trainEpoch makes, through the base class
    {
        auto twoSteps = [](nnet::Optimizer& optimizer, nnet::Parameter& weight) {
            for (int step = 0; step < 2; ++step) {
                weight.grad = nnet::Tensor({1, 1}, {1.0});
                optimizer.step();
            }
        };
        nnet::Parameter sgdWeight(nnet::Tensor({1, 1}, {0.0}));
        nnet::Parameter momentumWeight(nnet::Tensor({1, 1}, {0.0}));
        nnet::Parameter adamWeight(nnet::Tensor({1, 1}, {0.0}));
        nnet::SGDOptimizer sgd({&sgdWeight}, 0.1);
        nnet::MomentumOptimizer momentum({&momentumWeight}, 0.1);
        nnet::AdamOptimizer adam({&adamWeight}, 0.1);
        twoSteps(sgd, sgdWeight);
        twoSteps(momentum, momentumWeight);
        twoSteps(adam, adamWeight);
        tests.expectNear(sgdWeight.value.at({0, 0}), -0.2, tolerance, "an Optimizer& runs SGD's step");
        tests.expectNear(momentumWeight.value.at({0, 0}), -0.029, tolerance, "an Optimizer& runs momentum's step");
        tests.expectNear(adamWeight.value.at({0, 0}), -0.2, 1.0e-7, "an Optimizer& runs Adam's step");
    }

    return tests.finish();
}
