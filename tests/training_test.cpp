// Codex wrote these loss, backpropagation, and numerical gradient tests.

#include "test_utils.h"
#include "training.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace {



double calculateFixtureLoss(
    Network& model,
    const Trainer& trainer,
    const std::vector<double>& input,
    double target
) {
    const auto predictions = model.predict({input});
    return trainer.binaryCrossEntropy(predictions, {target});
}

double relativeError(double analytic, double numerical) {
    const double scale = std::max(
        1.0e-12,
        std::abs(analytic) + std::abs(numerical)
    );
    return std::abs(analytic - numerical) / scale;
}

std::string weightName(
    std::string_view prefix,
    std::size_t layer,
    std::size_t node,
    std::size_t input
) {
    std::ostringstream name;
    name << prefix << " layer " << layer
         << ", node " << node
         << ", input " << input;
    return name.str();
}

std::string biasName(
    std::string_view prefix,
    std::size_t layer,
    std::size_t node
) {
    std::ostringstream name;
    name << prefix << " layer " << layer
         << ", node " << node;
    return name.str();
}

} // namespace

void runTrainingTests(TestRunner& tests) {
    const double learningRate = 0.1;

    tests.section("Loss and gradients");

    Network model({2, 2, 1});
    configureGradientFixture(model);
    Trainer trainer(model, learningRate);

    const std::vector<double> input = gradientFixtureInput();
    constexpr double target = 1.0;
    constexpr double valueTolerance = 1.0e-12;

    const double loss = calculateFixtureLoss(model, trainer, input, target);
    tests.expectNear(
        loss,
        0.697899879671628,
        valueTolerance,
        "binary cross-entropy"
    );

    const std::vector<Layer> parametersBeforeBackprop = model.getNetwork();
    trainer.computeGradients(input, target);
    tests.expectTrue(
        parametersMatch(parametersBeforeBackprop, model.getNetwork()),
        "computeGradients() does not update parameters"
    );

    const auto& analyticWeights = trainer.getWeightGradients();
    const auto& analyticBiases = trainer.getDeltas();

    const std::vector<std::vector<std::vector<double>>> expectedWeights = {
        {
            {0.022496270128989, -0.014997513419326},
            {-0.015069614331961, 0.010046409554641}
        },
        {
            {-0.268739667946555, -0.253697125559285}
        }
    };
    const std::vector<std::vector<double>> expectedBiases = {
        {0.037493783548315, -0.025116023886602},
        {-0.502370711454233}
    };

    const bool gradientShapesAreCorrect =
        analyticWeights.size() == expectedWeights.size() &&
        analyticBiases.size() == expectedBiases.size() &&
        analyticWeights[0].size() == expectedWeights[0].size() &&
        analyticWeights[1].size() == expectedWeights[1].size() &&
        analyticBiases[0].size() == expectedBiases[0].size() &&
        analyticBiases[1].size() == expectedBiases[1].size();

    tests.expectTrue(gradientShapesAreCorrect, "gradient shapes match parameter shapes");
    if (!gradientShapesAreCorrect) {
        return;
    }

    for (std::size_t layer = 0; layer < expectedWeights.size(); ++layer) {
        for (std::size_t node = 0; node < expectedWeights[layer].size(); ++node) {
            if (analyticWeights[layer][node].size() != expectedWeights[layer][node].size()) {
                tests.expectTrue(false, weightName("gradient width", layer, node, 0));
                return;
            }
            for (std::size_t inputIndex = 0;
                 inputIndex < expectedWeights[layer][node].size();
                 ++inputIndex) {
                tests.expectNear(
                    analyticWeights[layer][node][inputIndex],
                    expectedWeights[layer][node][inputIndex],
                    valueTolerance,
                    weightName("analytic weight gradient", layer, node, inputIndex)
                );
            }
        }
    }

    for (std::size_t layer = 0; layer < expectedBiases.size(); ++layer) {
        for (std::size_t node = 0; node < expectedBiases[layer].size(); ++node) {
            tests.expectNear(
                analyticBiases[layer][node],
                expectedBiases[layer][node],
                valueTolerance,
                biasName("analytic bias gradient", layer, node)
            );
        }
    }

    tests.section("Numerical gradient checking");
    constexpr double epsilon = 1.0e-5;
    constexpr double maximumRelativeError = 1.0e-5;
    const std::vector<Layer> parametersBeforeNumericalCheck = model.getNetwork();
    auto& layers = model.getNetwork();

    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
        for (std::size_t node = 0; node < layers[layer].weights.size(); ++node) {
            for (std::size_t inputIndex = 0;
                 inputIndex < layers[layer].weights[node].size();
                 ++inputIndex) {
                double& weight = layers[layer].weights[node][inputIndex];
                const double original = weight;

                weight = original + epsilon;
                const double lossPlus = calculateFixtureLoss(model, trainer, input, target);

                weight = original - epsilon;
                const double lossMinus = calculateFixtureLoss(model, trainer, input, target);

                weight = original;
                const double numerical = (lossPlus - lossMinus) / (2.0 * epsilon);
                const double error = relativeError(
                    analyticWeights[layer][node][inputIndex],
                    numerical
                );

                tests.expectTrue(
                    error < maximumRelativeError,
                    weightName("numerical weight gradient", layer, node, inputIndex)
                );
            }
        }
    }

    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
        for (std::size_t node = 0; node < layers[layer].biases.size(); ++node) {
            double& bias = layers[layer].biases[node];
            const double original = bias;

            bias = original + epsilon;
            const double lossPlus = calculateFixtureLoss(model, trainer, input, target);

            bias = original - epsilon;
            const double lossMinus = calculateFixtureLoss(model, trainer, input, target);

            bias = original;
            const double numerical = (lossPlus - lossMinus) / (2.0 * epsilon);
            const double error = relativeError(
                analyticBiases[layer][node],
                numerical
            );

            tests.expectTrue(
                error < maximumRelativeError,
                biasName("numerical bias gradient", layer, node)
            );
        }
    }

    tests.section("SGD optimizer updates weights and biases");
    trainer.sgdOptimizer();
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        for (std::size_t node = 0; node < layers[layer].weights.size(); node++) {
            for (std::size_t input = 0; input < layers[layer].weights[node].size(); input++) {
                double expected = parametersBeforeNumericalCheck[layer].weights[node][input] - analyticWeights[layer][node][input] * learningRate;
                tests.expectNear(layers[layer].weights[node][input], expected, 1e-19, weightName("sgdOptimizer() updates weights", layer, node, input));
            }
            double expectedBias = parametersBeforeNumericalCheck[layer].biases[node] - analyticBiases[layer][node] * learningRate;
            tests.expectNear(layers[layer].biases[node], expectedBias, 1e-19, biasName("sgdOptimizer() updates biases", layer, node));
        }
    }

}
