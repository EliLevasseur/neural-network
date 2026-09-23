#include "test_utils.h"
#include "training.h"
#include "nnet/core/tensor.h"
#include "nnet/core/tensor_ops.h"
#include "nnet/nn/dense.h"

#include <vector>

// Vocabulary used below, since the usual maths shorthand hides what these are:
//
//   preActivation  the weighted sum plus the bias, BEFORE sigmoid is applied.
//   activation     the same values AFTER sigmoid. This is what the next layer
//                  receives as its input.
//   gradient       how much the loss changes per unit change in that value.
//                  Exactly the same quantity the legacy Trainer calls a
//                  "delta" for biases and a "weight gradient" for weights.

// Values saved by one forward pass, for that same pass's backward to use.
// Created by the call and destroyed with it, so no layer caches anything.
struct ForwardTrace {
    std::vector<nnet::Tensor> activations;     // the input, then one per layer
    std::vector<nnet::Tensor> preActivations;  // one per layer
};

// These two live here rather than in nnet/ on purpose. They hard-code sigmoid
// after every layer and binary cross entropy as the loss, and both of those
// are choices belonging to one specific model, not capabilities of the
// framework. Promote whatever turns out to be genuinely general only once a
// second model wants it.
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
    // Start from the gradient of the loss with respect to the prediction.
    nnet::Tensor gradient =
        nnet::binaryCrossEntropyGrad(trace.activations.back(), target);

    // An unsigned index is never below zero, so it is decremented at the top
    // rather than in the loop header. The legacy Trainer walks back the same
    // way for the same reason.
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

void runParityTests(TestRunner& tests) {
    tests.section("REFERENCE / TENSOR PARITY");

    const double target = 1.0;
    const double tolerance = 1.0e-12;
    const double learningRate = 0.09;

    // ---- 1. Ask the legacy model for the trusted answer -----------------
    Network referenceModel({2, 2, 1});
    configureGradientFixture(referenceModel);

    Trainer referenceTrainer(referenceModel, learningRate);
    referenceTrainer.computeGradients(gradientFixtureInput(), target);

    const auto& referenceWeightGradients = referenceTrainer.getWeightGradients();

    // The legacy trainer's "deltas" are its bias gradients: sgdOptimizer()
    // subtracts them straight from the biases.
    const auto& referenceBiasGradients = referenceTrainer.getDeltas();

    // ---- 2. Same model, built from Dense layers ----------
    // Dense starts from random weights, so they are overwritten with the
    // fixture's fixed values. The legacy Network stores weights[node][input]
    // and a Dense stores [input][node], so each matrix is the transpose of
    // the fixture's layout.
    const nnet::Tensor inputRow({1, 2}, {0.6, -0.4});
    const nnet::Tensor targetTensor({1, 1}, {target});

    std::vector<nnet::Dense> layers;
    layers.reserve(2);
    layers.emplace_back(2, 2);
    layers.emplace_back(2, 1);

    layers[0].weight.value = nnet::Tensor({2, 2}, { 0.1, 0.4,
                                                   -0.2, 0.3});
    layers[0].bias.value = nnet::Tensor({2}, {0.0, -0.1});
    layers[1].weight.value = nnet::Tensor({2, 1}, {-0.3,
                                                    0.2});
    layers[1].bias.value = nnet::Tensor({1}, {0.05});

    // Nothing below knows how many layers there are.
    const ForwardTrace trace = forwardPass(layers, inputRow);
    backwardPass(layers, trace, targetTensor);

    // ---- 3. Compare every gradient --------------------------------------
    // The legacy index is [node][input]. The Dense index is {input, node},
    // because of the transposed layout above. Getting this pair backwards is
    // the most likely way for this test to fail for a non-maths reason.
    for (std::size_t node = 0; node < 2; node++) {
        for (std::size_t input = 0; input < 2; input++) {
            tests.expectNear(
                layers[0].weight.grad.at({input, node}),
                referenceWeightGradients[0][node][input],
                tolerance,
                "hidden weight gradient matches the reference model");
        }
    }

    for (std::size_t input = 0; input < 2; input++) {
        tests.expectNear(
            layers[1].weight.grad.at({input, 0}),
            referenceWeightGradients[1][0][input],
            tolerance,
            "output weight gradient matches the reference model");
    }

    for (std::size_t node = 0; node < 2; node++) {
        tests.expectNear(
            layers[0].bias.grad.at({node}),
            referenceBiasGradients[0][node],
            tolerance,
            "hidden bias gradient matches the reference model");
    }

    tests.expectNear(
        layers[1].bias.grad.at({0}),
        referenceBiasGradients[1][0],
        tolerance,
        "output bias gradient matches the reference model");

    // ---- 4. Apply one optimizer step and compare the updated values -----
    // Every layer is asked for its own parameters, so this names neither a
    // weight nor a bias and works unchanged for any number of layers.
    for (nnet::Dense& layer : layers) {
        for (nnet::Parameter* parameter : layer.parameters()) {
            parameter->value = parameter->value - (parameter->grad * learningRate);
        }
    }

    // The legacy optimizer rewrites the model in place, so the reference
    // values have to be read AFTER this call.
    referenceTrainer.sgdOptimizer();
    const auto& updatedReferenceLayers = referenceModel.getNetwork();

    for (std::size_t node = 0; node < 2; node++) {
        for (std::size_t input = 0; input < 2; input++) {
            tests.expectNear(
                layers[0].weight.value.at({input, node}),
                updatedReferenceLayers[0].weights[node][input],
                tolerance,
                "hidden weight after one SGD step matches the reference model");
        }
    }

    for (std::size_t input = 0; input < 2; input++) {
        tests.expectNear(
            layers[1].weight.value.at({input, 0}),
            updatedReferenceLayers[1].weights[0][input],
            tolerance,
            "output weight after one SGD step matches the reference model");
    }

    for (std::size_t node = 0; node < 2; node++) {
        tests.expectNear(
            layers[0].bias.value.at({node}),
            updatedReferenceLayers[0].biases[node],
            tolerance,
            "hidden bias after one SGD step matches the reference model");
    }

    tests.expectNear(
        layers[1].bias.value.at({0}),
        updatedReferenceLayers[1].biases[0],
        tolerance,
        "output bias after one SGD step matches the reference model");
    // ---- 5. The same loop, three layers deep ----------------------------
    // Nothing in forwardPass or backwardPass knows the layer count, so a
    // deeper network should need no new code. This builds a {2, 3, 2, 1}
    // model on both sides from one shared set of deterministic weights and
    // compares every gradient.
    tests.section("REFERENCE / TENSOR PARITY (THREE LAYERS)");

    const std::vector<std::size_t> deepShape = {2, 3, 2, 1};

    Network deepReference(deepShape);
    Trainer deepTrainer(deepReference, learningRate);
    auto& deepReferenceLayers = deepReference.getNetwork();

    std::vector<nnet::Dense> deepLayers;
    deepLayers.reserve(deepShape.size() - 1);
    for (std::size_t i = 0; i + 1 < deepShape.size(); i++) {
        deepLayers.emplace_back(deepShape[i], deepShape[i + 1]);
    }

    // Fill both models from the same formula, so neither depends on the
    // unseeded random initialiser.
    for (std::size_t layer = 0; layer < deepLayers.size(); layer++) {
        const std::size_t inputCount = deepShape[layer];
        const std::size_t nodeCount = deepShape[layer + 1];

        for (std::size_t node = 0; node < nodeCount; node++) {
            for (std::size_t input = 0; input < inputCount; input++) {
                const double weightValue =
                    0.05 * static_cast<double>((layer * 5 + node * 3 + input * 2) % 11) - 0.25;
                deepReferenceLayers[layer].weights[node][input] = weightValue;
                deepLayers[layer].weight.value.at({input, node}) = weightValue;
            }
            const double biasValue =
                0.02 * static_cast<double>((layer * 3 + node) % 7) - 0.06;
            deepReferenceLayers[layer].biases[node] = biasValue;
            deepLayers[layer].bias.value.at({node}) = biasValue;
        }
    }

    deepTrainer.computeGradients(gradientFixtureInput(), target);
    const auto& deepWeightGradients = deepTrainer.getWeightGradients();
    const auto& deepBiasGradients = deepTrainer.getDeltas();

    const ForwardTrace deepTrace = forwardPass(deepLayers, inputRow);
    backwardPass(deepLayers, deepTrace, targetTensor);

    for (std::size_t layer = 0; layer < deepLayers.size(); layer++) {
        const std::size_t inputCount = deepShape[layer];
        const std::size_t nodeCount = deepShape[layer + 1];

        for (std::size_t node = 0; node < nodeCount; node++) {
            for (std::size_t input = 0; input < inputCount; input++) {
                tests.expectNear(
                    deepLayers[layer].weight.grad.at({input, node}),
                    deepWeightGradients[layer][node][input],
                    tolerance,
                    "three-layer weight gradient matches the reference model");
            }
            tests.expectNear(
                deepLayers[layer].bias.grad.at({node}),
                deepBiasGradients[layer][node],
                tolerance,
                "three-layer bias gradient matches the reference model");
        }
    }

}
