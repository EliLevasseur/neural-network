// Codex wrote these deterministic forward-pass tests.

#include "test_utils.h"

#include <vector>

void runNetworkTests(TestRunner& tests) {
    tests.section("Network forward pass");

    Network model({2, 2, 1});
    configureGradientFixture(model);
    const std::vector<double> input = gradientFixtureInput();

    const auto& activations = model.getActivations(input);
    const bool activationShapeIsCorrect =
        activations.size() == 3 &&
        activations[0].size() == 2 &&
        activations[1].size() == 2 &&
        activations[2].size() == 1;

    tests.expectTrue(activationShapeIsCorrect, "activation shapes are {2}, {2}, {1}");
    if (!activationShapeIsCorrect) {
        return;
    }

    constexpr double tolerance = 1.0e-12;
    tests.expectNear(
        activations[1][0],
        0.534942945158215,
        tolerance,
        "first hidden activation"
    );
    tests.expectNear(
        activations[1][1],
        0.504999833340000,
        tolerance,
        "second hidden activation"
    );
    tests.expectNear(
        activations[2][0],
        0.497629288545767,
        tolerance,
        "output activation"
    );

    const auto predictions = model.predict({input});
    tests.expectEqual(predictions.size(), std::size_t{1}, "prediction count");
    if (!predictions.empty()) {
        tests.expectNear(
            predictions[0],
            0.497629288545767,
            tolerance,
            "predict() matches the expected output"
        );
    }
}
