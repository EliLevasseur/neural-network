#include "test_utils.h"
#include "training.h"
#include "nnet/core/tensor.h"
#include "nnet/core/tensor_ops.h"
#include "nnet/core/autograd/backward.h"
#include "nnet/core/autograd/operations.h"
#include "nnet/nn/dense.h"
#include "nnet/nn/sequential.h"

#include <memory>
#include <utility>
#include <vector>

// Three ways of computing the same gradients are compared in this file:
//
//   1. the legacy Network and Trainer, the original oracle;
//   2. a manual backward built from the M2 gradient functions on plain
//      Tensors, touching neither Dense nor autograd;
//   3. autograd, through Dense, Sigmoid, and Sequential.
//
// Each one is checked against the legacy model. The manual path stays so
// autograd always has a second, independent reference beside the original.
//
// Vocabulary:
//   preActivation  the weighted sum plus the bias, BEFORE sigmoid is applied.
//   activation     the same values AFTER sigmoid, fed to the next layer.
//   gradient       how much the loss changes per unit change in that value.

namespace {

    // One layer's numbers, stored the way a Dense stores them: [input][node].
    struct LayerWeights {
        nnet::Tensor weight;
        nnet::Tensor bias;
    };

    struct LayerGradients {
        nnet::Tensor weight;
        nnet::Tensor bias;
    };

    struct ForwardTrace {
        std::vector<nnet::Tensor> activations;     // the input, then one per layer
        std::vector<nnet::Tensor> preActivations;  // one per layer
    };

    // ---- the manual oracle ------------------------------------------------
    // Plain Tensors and the gradient functions from M2, nothing else.

    ForwardTrace manualForward(const std::vector<LayerWeights>& layers,
                               const nnet::Tensor& input) {
        ForwardTrace trace;
        trace.activations.push_back(input);
        for (const LayerWeights& layer : layers) {
            nnet::Tensor preActivation = trace.activations.back() * layer.weight;
            nnet::addBias(preActivation, layer.bias);
            trace.preActivations.push_back(preActivation);
            trace.activations.push_back(nnet::sigmoid(preActivation));
        }
        return trace;
    }

    std::vector<LayerGradients> manualBackward(const std::vector<LayerWeights>& layers,
                                               const ForwardTrace& trace,
                                               const nnet::Tensor& target) {
        std::vector<LayerGradients> gradients(layers.size(),
            LayerGradients{nnet::Tensor({1}, {0.0}), nnet::Tensor({1}, {0.0})});

        nnet::Tensor gradient =
            nnet::binaryCrossEntropyGrad(trace.activations.back(), target);

        for (std::size_t index = layers.size(); index > 0;) {
            --index;
            gradient.inplaceMultiplication(
                nnet::sigmoidDerivitive(trace.preActivations[index]));
            gradients[index].weight =
                nnet::matmulGradWeight(trace.activations[index], gradient);
            gradients[index].bias = nnet::addBiasGrad(gradient, layers[index].bias);
            gradient = nnet::matmulGradInput(gradient, layers[index].weight);
        }
        return gradients;
    }

    // ---- the autograd model -------------------------------------------------

    // Sequential owns the layers. The Dense pointers only observe them, so the
    // test can read each layer's gradient afterwards.
    struct AutogradModel {
        std::unique_ptr<nnet::Sequential> model;
        std::vector<nnet::Dense*> layers;
    };

    AutogradModel buildAutogradModel(const std::vector<LayerWeights>& weights) {
        AutogradModel result;
        std::vector<std::unique_ptr<nnet::Unary_Module>> components;
        for (const LayerWeights& layer : weights) {
            auto dense = std::make_unique<nnet::Dense>(
                layer.weight.shape()[0], layer.weight.shape()[1]);
            dense->weight.value = layer.weight;
            dense->bias.value = layer.bias;
            result.layers.push_back(dense.get());
            components.push_back(std::move(dense));
            components.push_back(std::make_unique<nnet::Sigmoid>());
        }
        result.model = std::make_unique<nnet::Sequential>(std::move(components));
        return result;
    }

    // ---- helpers shared by every section ----------------------------------

    // Copies a legacy Network's weights into Dense layout, transposing each
    // matrix from [node][input] to [input][node].
    std::vector<LayerWeights> weightsFromReference(Network& reference) {
        std::vector<LayerWeights> result;
        for (const Layer& layer : reference.getNetwork()) {
            const std::size_t nodes = layer.weights.size();
            const std::size_t inputs = layer.weights[0].size();
            std::vector<double> weight(inputs * nodes);
            for (std::size_t node = 0; node < nodes; ++node) {
                for (std::size_t input = 0; input < inputs; ++input) {
                    weight[input * nodes + node] = layer.weights[node][input];
                }
            }
            result.push_back({nnet::Tensor({inputs, nodes}, weight),
                              nnet::Tensor({nodes}, layer.biases)});
        }
        return result;
    }

    // The reference exposes activations, not preactivations, so each weighted
    // sum is rebuilt from its own layout and saved activations.
    void compareForwardTrace(TestRunner& tests, Network& reference,
                             const ForwardTrace& trace,
                             const std::vector<double>& input, double target) {
        const auto activations = reference.getActivations(input);
        const auto& layers = reference.getNetwork();
        constexpr double tolerance = 1.0e-12;
        for (std::size_t layer = 0; layer < layers.size(); ++layer) {
            for (std::size_t node = 0; node < layers[layer].weights.size(); ++node) {
                double weightedSum = 0.0;
                for (std::size_t feature = 0; feature < activations[layer].size(); ++feature) {
                    weightedSum += activations[layer][feature] * layers[layer].weights[node][feature];
                }
                weightedSum += layers[layer].biases[node];
                tests.expectNear(trace.preActivations[layer].at({0, node}), weightedSum,
                    tolerance, "preactivation matches the reference weighted sum");
                tests.expectNear(trace.activations[layer + 1].at({0, node}),
                    activations[layer + 1][node], tolerance,
                    "activation matches the reference model");
            }
        }
        Trainer referenceLoss(reference, 0.1);
        tests.expectNear(
            nnet::binaryCrossEntropy(trace.activations.back(), nnet::Tensor({1, 1}, {target})),
            referenceLoss.binaryCrossEntropy(activations.back(), {target}), tolerance,
            "BCE matches the reference loss");
    }

    // Compares every weight and bias gradient against the legacy trainer.
    // The legacy index is [node][input]; ours is {input, node}.
    void compareGradients(TestRunner& tests, Trainer& trainer,
                          const std::vector<LayerGradients>& gradients,
                          const char* weightName, const char* biasName) {
        const auto& weightGradients = trainer.getWeightGradients();
        const auto& biasGradients = trainer.getDeltas();
        for (std::size_t layer = 0; layer < gradients.size(); ++layer) {
            const std::size_t nodes = biasGradients[layer].size();
            for (std::size_t node = 0; node < nodes; ++node) {
                for (std::size_t input = 0; input < weightGradients[layer][node].size(); ++input) {
                    tests.expectNear(gradients[layer].weight.at({input, node}),
                        weightGradients[layer][node][input], 1.0e-12, weightName);
                }
                tests.expectNear(gradients[layer].bias.at({node}),
                    biasGradients[layer][node], 1.0e-12, biasName);
            }
        }
    }

    std::vector<LayerGradients> gradientsOf(const AutogradModel& autograd) {
        std::vector<LayerGradients> result;
        for (const nnet::Dense* layer : autograd.layers) {
            result.push_back({layer->weight.grad, layer->bias.grad});
        }
        return result;
    }

    // Runs one autograd training step's gradient half: clear, forward, backward.
    nnet::Value autogradStep(AutogradModel& autograd,
                             const nnet::Tensor& input, const nnet::Tensor& target) {
        autograd.model->zeroGrad();
        const nnet::Value prediction = autograd.model->forward(nnet::makeLeaf(input));
        nnet::backward(nnet::binaryCrossEntropy(prediction, nnet::makeLeaf(target)));
        return prediction;
    }

    // ---- one Dense, two invocations, through autograd ---------------------
    void checkReusedDense(TestRunner& tests) {
        tests.section("ONE DENSE, TWO INVOCATIONS (AUTOGRAD)");

        nnet::Dense shared(1, 1);
        shared.weight.value.at({0, 0}) = 0.7;
        shared.bias.value.at({0}) = -0.2;
        const nnet::Tensor target({1, 1}, {1.0});

        // The same layer object runs twice in one graph. Each call makes its
        // own leaves, and both point at the same two Parameters.
        shared.zeroGrad();
        const nnet::Value hidden = nnet::sigmoid(shared.forward(
            nnet::makeLeaf(nnet::Tensor({1, 1}, {0.4}))));
        const nnet::Value prediction = nnet::sigmoid(shared.forward(hidden));
        nnet::backward(nnet::binaryCrossEntropy(prediction, nnet::makeLeaf(target)));

        // Two separate legacy layers with identical numbers. Their two partial
        // derivatives must sum to the shared parameter's derivative.
        Network reference({1, 1, 1});
        for (auto& layer : reference.getNetwork()) {
            layer.weights[0][0] = 0.7;
            layer.biases[0] = -0.2;
        }
        Trainer trainer(reference, 0.1);
        trainer.computeGradients({0.4}, 1.0);

        tests.expectNear(prediction->data.at({0, 0}),
            reference.getActivations({0.4}).back()[0], 1.0e-12,
            "reused Dense forward matches two equal legacy layers");
        tests.expectNear(shared.weight.grad.at({0, 0}),
            trainer.getWeightGradients()[0][0][0] + trainer.getWeightGradients()[1][0][0],
            1.0e-12, "shared weight receives both legacy gradient contributions");
        tests.expectNear(shared.bias.grad.at({0}),
            trainer.getDeltas()[0][0] + trainer.getDeltas()[1][0],
            1.0e-12, "shared bias receives both legacy gradient contributions");

        // Whole-computation finite differences, independent of every other path.
        auto loss = [&](double weight, double bias) {
            nnet::Tensor h = nnet::Tensor({1, 1}, {0.4}) * nnet::Tensor({1, 1}, {weight});
            nnet::addBias(h, nnet::Tensor({1}, {bias}));
            h = nnet::sigmoid(h);
            nnet::Tensor out = h * nnet::Tensor({1, 1}, {weight});
            nnet::addBias(out, nnet::Tensor({1}, {bias}));
            return nnet::binaryCrossEntropy(nnet::sigmoid(out), target);
        };
        constexpr double step = 1.0e-6;
        tests.expectNear(shared.weight.grad.at({0, 0}),
            (loss(0.7 + step, -0.2) - loss(0.7 - step, -0.2)) / (2.0 * step), 1.0e-8,
            "shared weight gradient matches whole-computation finite differences");
        tests.expectNear(shared.bias.grad.at({0}),
            (loss(0.7, -0.2 + step) - loss(0.7, -0.2 - step)) / (2.0 * step), 1.0e-8,
            "shared bias gradient matches whole-computation finite differences");
    }
}

void runParityTests(TestRunner& tests) {
    tests.section("REFERENCE / MANUAL / AUTOGRAD PARITY");

    const double target = 1.0;
    const double tolerance = 1.0e-12;
    const double learningRate = 0.09;
    const nnet::Tensor inputRow({1, 2}, {0.6, -0.4});
    const nnet::Tensor targetTensor({1, 1}, {target});

    // ---- 1. The legacy model's answer -------------------------------------
    Network referenceModel({2, 2, 1});
    configureGradientFixture(referenceModel);
    const std::vector<LayerWeights> weights = weightsFromReference(referenceModel);

    Trainer referenceTrainer(referenceModel, learningRate);
    referenceTrainer.computeGradients(gradientFixtureInput(), target);

    // ---- 2. The manual oracle ----------------------------------------------
    const ForwardTrace trace = manualForward(weights, inputRow);
    compareForwardTrace(tests, referenceModel, trace, gradientFixtureInput(), target);
    const std::vector<LayerGradients> manualGradients =
        manualBackward(weights, trace, targetTensor);
    compareGradients(tests, referenceTrainer, manualGradients,
        "manual weight gradient matches the reference model",
        "manual bias gradient matches the reference model");

    // ---- 3. Autograd, through Dense, Sigmoid, and Sequential ---------------
    AutogradModel autograd = buildAutogradModel(weights);
    const nnet::Value prediction = autogradStep(autograd, inputRow, targetTensor);

    tests.expectTrue(prediction->data.shape() == nnet::Tensor::Shape{1, 1},
        "autograd prediction has the expected shape");
    tests.expectNear(prediction->data.at({0, 0}),
        referenceModel.getActivations(gradientFixtureInput()).back().at(0), tolerance,
        "autograd prediction matches the reference model");

    const std::vector<LayerGradients> autogradGradients = gradientsOf(autograd);
    compareGradients(tests, referenceTrainer, autogradGradients,
        "autograd weight gradient matches the reference model",
        "autograd bias gradient matches the reference model");

    // Autograd against the manual oracle directly, element by element.
    bool autogradMatchesManual = true;
    for (std::size_t layer = 0; layer < weights.size(); ++layer) {
        for (std::size_t i = 0; i < manualGradients[layer].weight.numel(); ++i) {
            autogradMatchesManual = autogradMatchesManual &&
                std::abs(autogradGradients[layer].weight.getData()[i] -
                         manualGradients[layer].weight.getData()[i]) <= tolerance;
        }
        for (std::size_t i = 0; i < manualGradients[layer].bias.numel(); ++i) {
            autogradMatchesManual = autogradMatchesManual &&
                std::abs(autogradGradients[layer].bias.getData()[i] -
                         manualGradients[layer].bias.getData()[i]) <= tolerance;
        }
    }
    tests.expectTrue(autogradMatchesManual,
        "autograd gradients equal the manual backward's gradients");

    // ---- 4. One optimizer step on the autograd model -----------------------
    // The model is asked for its own parameters, so this never names a layer.
    for (nnet::Parameter* parameter : autograd.model->parameters()) {
        parameter->value = parameter->value - (parameter->grad * learningRate);
    }
    referenceTrainer.sgdOptimizer();
    const std::vector<LayerWeights> updated = weightsFromReference(referenceModel);

    for (std::size_t layer = 0; layer < updated.size(); ++layer) {
        for (std::size_t i = 0; i < updated[layer].weight.numel(); ++i) {
            tests.expectNear(autograd.layers[layer]->weight.value.getData()[i],
                updated[layer].weight.getData()[i], tolerance,
                "weight after one SGD step matches the reference model");
        }
        for (std::size_t i = 0; i < updated[layer].bias.numel(); ++i) {
            tests.expectNear(autograd.layers[layer]->bias.value.getData()[i],
                updated[layer].bias.getData()[i], tolerance,
                "bias after one SGD step matches the reference model");
        }
    }

    // A second step must start from zero again rather than adding onto the
    // first step's gradients.
    autogradStep(autograd, inputRow, targetTensor);
    Trainer secondTrainer(referenceModel, learningRate);
    secondTrainer.computeGradients(gradientFixtureInput(), target);
    compareGradients(tests, secondTrainer, gradientsOf(autograd),
        "second-step weight gradient matches the reference model",
        "second-step bias gradient matches the reference model");

    // ---- 5. Three layers deep ------------------------------------------------
    tests.section("REFERENCE / MANUAL / AUTOGRAD PARITY (THREE LAYERS)");

    const std::vector<std::size_t> deepShape = {2, 3, 2, 1};
    Network deepReference(deepShape);
    auto& deepReferenceLayers = deepReference.getNetwork();

    // Filled from a formula, so nothing depends on the unseeded initialiser.
    for (std::size_t layer = 0; layer + 1 < deepShape.size(); layer++) {
        for (std::size_t node = 0; node < deepShape[layer + 1]; node++) {
            for (std::size_t input = 0; input < deepShape[layer]; input++) {
                deepReferenceLayers[layer].weights[node][input] =
                    0.05 * static_cast<double>((layer * 5 + node * 3 + input * 2) % 11) - 0.25;
            }
            deepReferenceLayers[layer].biases[node] =
                0.02 * static_cast<double>((layer * 3 + node) % 7) - 0.06;
        }
    }
    const std::vector<LayerWeights> deepWeights = weightsFromReference(deepReference);

    Trainer deepTrainer(deepReference, learningRate);
    deepTrainer.computeGradients(gradientFixtureInput(), target);

    const ForwardTrace deepTrace = manualForward(deepWeights, inputRow);
    compareForwardTrace(tests, deepReference, deepTrace, gradientFixtureInput(), target);
    compareGradients(tests, deepTrainer, manualBackward(deepWeights, deepTrace, targetTensor),
        "three-layer manual weight gradient matches the reference model",
        "three-layer manual bias gradient matches the reference model");

    AutogradModel deepAutograd = buildAutogradModel(deepWeights);
    autogradStep(deepAutograd, inputRow, targetTensor);
    compareGradients(tests, deepTrainer, gradientsOf(deepAutograd),
        "three-layer autograd weight gradient matches the reference model",
        "three-layer autograd bias gradient matches the reference model");

    checkReusedDense(tests);
}
