#ifndef TEST_UTILS_H
#define TEST_UTILS_H

// Codex wrote this test harness and the initial deterministic test fixture.

#include "network.h"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

class TestRunner {
public:
    void section(std::string_view name) const {
        std::cout << "\n[ " << name << " ]\n";
    }

    void expectTrue(bool condition, std::string_view name) {
        if (condition) {
            ++passed;
            std::cout << "  PASS  " << name << '\n';
            return;
        }

        ++failed;
        std::cout << "  FAIL  " << name << '\n';
    }

    template <typename Actual, typename Expected>
    void expectEqual(const Actual& actual, const Expected& expected, std::string_view name) {
        if (actual == expected) {
            ++passed;
            std::cout << "  PASS  " << name << '\n';
            return;
        }

        ++failed;
        std::cout << "  FAIL  " << name
                  << " (expected " << expected
                  << ", got " << actual << ")\n";
    }

    void expectNear(
        double actual,
        double expected,
        double tolerance,
        std::string_view name
    ) {
        const double difference = std::abs(actual - expected);
        if (std::isfinite(actual) && difference <= tolerance) {
            ++passed;  
            std::cout << "  PASS  " << name << '\n';
            return;
        }

        ++failed;
        std::cout << std::setprecision(15)
                  << "  FAIL  " << name
                  << " (expected " << expected
                  << ", got " << actual
                  << ", difference " << difference << ")\n";
    }

    int finish() const {
        std::cout << "\nTest summary: " << passed << " passed, "
                  << failed << " failed.\n";
        return failed == 0 ? 0 : 1;
    }

private:
    std::size_t passed = 0;
    std::size_t failed = 0;
};

inline std::vector<double> gradientFixtureInput() {
    return {0.6, -0.4};
}

inline void configureGradientFixture(Network& model) {
    auto& layers = model.getNetwork();
    if (layers.size() != 2 ||
        layers[0].weights.size() != 2 ||
        layers[1].weights.size() != 1) {
        throw std::logic_error("Gradient fixture requires a {2, 2, 1} network.");
    }

    layers[0].weights = {
        {0.1, -0.2},
        {0.4, 0.3}
    };
    layers[0].biases = {0.0, -0.1};

    layers[1].weights = {
        {-0.3, 0.2}
    };
    layers[1].biases = {0.05};
}

inline bool parametersMatch(
    const std::vector<Layer>& expected,
    const std::vector<Layer>& actual
) {
    if (expected.size() != actual.size()) {
        return false;
    }

    for (std::size_t layer = 0; layer < expected.size(); ++layer) {
        if (expected[layer].weights != actual[layer].weights ||
            expected[layer].biases != actual[layer].biases) {
            return false;
        }
    }

    return true;
}

#endif
