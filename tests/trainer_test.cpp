#include "test_utils.h"
#include "nnet/train/trainer.h"
#include "nnet/nn/dense.h"
#include "nnet/nn/sequential.h"
#include "nnet/core/autograd/backward.h"
#include "nnet/core/autograd/operations.h"
#include "optim/sgd.h"

#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
    // A small Dense, ReLU, Dense network. Two calls with the same seed build
    // two separate models that start from identical weights.
    std::unique_ptr<nnet::Sequential> buildModel(std::size_t inputs, std::size_t hidden,
                                                 std::size_t outputs, unsigned seed) {
        std::mt19937 generator(seed);
        std::vector<std::unique_ptr<nnet::Unary_Module>> components;
        components.push_back(std::make_unique<nnet::Dense>(inputs, hidden, generator));
        components.push_back(std::make_unique<nnet::ReLU>());
        components.push_back(std::make_unique<nnet::Dense>(hidden, outputs, generator));
        return std::make_unique<nnet::Sequential>(std::move(components));
    }

    // Written so a NaN counts as different.
    bool sameParameters(nnet::Module& first, nnet::Module& second, double tolerance) {
        const std::vector<nnet::Parameter*> firstParameters = first.parameters();
        const std::vector<nnet::Parameter*> secondParameters = second.parameters();
        if (firstParameters.size() != secondParameters.size()) {
            return false;
        }
        for (std::size_t i = 0; i < firstParameters.size(); ++i) {
            const std::vector<double>& firstValues = firstParameters[i]->value.getData();
            const std::vector<double>& secondValues = secondParameters[i]->value.getData();
            if (firstValues.size() != secondValues.size()) {
                return false;
            }
            for (std::size_t j = 0; j < firstValues.size(); ++j) {
                if (!(std::abs(firstValues[j] - secondValues[j]) <= tolerance)) {
                    return false;
                }
            }
        }
        return true;
    }
}

void runTrainerTests(TestRunner& tests) {
    tests.section("TRAINER");

    // Five examples, two features, two classes written as one-hot rows.
    const nnet::Tensor inputs({5, 2}, {0.5, -0.2, -0.7, 0.4, 0.9, 0.8, -0.3, -0.6, 0.1, 0.3});
    const nnet::Tensor targets({5, 2}, {1, 0, 0, 1, 1, 0, 0, 1, 1, 0});

    // ---- takeRows ----------------------------------------------------------
    {
        const nnet::Tensor data({3, 2}, {1, 2, 3, 4, 5, 6});
        const nnet::Tensor picked = nnet::takeRows(data, {2, 0});
        tests.expectTrue(picked.shape() == nnet::Tensor::Shape{2, 2} &&
            picked.getData() == std::vector<double>{5, 6, 1, 2},
            "takeRows copies the listed rows in the order given");

        bool rejected = false;
        try {
            nnet::takeRows(data, {3});
        } catch (const std::out_of_range&) {
            rejected = true;
        }
        tests.expectTrue(rejected, "takeRows rejects a row index past the last row");
    }

    // ---- one batch holding every row is plain gradient descent -------------
    // Two epochs with all five rows in one batch must match two steps done by
    // hand. This also proves gradients are cleared between steps: without
    // zeroGrad, the second step would still carry the first step's gradient.
    {
        auto trained = buildModel(2, 3, 2, 21);
        auto byHand = buildModel(2, 3, 2, 21);
        std::mt19937 shuffleGenerator(4);
        for (int epoch = 0; epoch < 2; ++epoch) {
            nnet::trainEpoch(*trained, inputs, targets, 5, 0.3, nnet::softmaxCrossEntropy, shuffleGenerator);
        }

        std::vector<nnet::Parameter*> parameters = byHand->parameters();
        for (int step = 0; step < 2; ++step) {
            byHand->zeroGrad();
            const nnet::Value scores = byHand->forward(nnet::makeLeaf(inputs));
            nnet::backward(nnet::softmaxCrossEntropy(scores, nnet::makeLeaf(targets)));
            nnet::sgdOptimizer(parameters, 0.3);
        }
        tests.expectTrue(sameParameters(*trained, *byHand, 1.0e-12),
            "two full-batch epochs match two gradient descent steps done by hand");
    }

    // ---- every row counts once, and a short last batch is weighted fairly --
    // With a learning rate of 0 the weights never move, so the epoch's
    // average loss must equal the loss over all five rows at once, even
    // though the batches hold 2, 2 and 1 rows.
    {
        auto model = buildModel(2, 3, 2, 21);
        std::mt19937 shuffleGenerator(4);
        const double epochLoss =
            nnet::trainEpoch(*model, inputs, targets, 2, 0.0, nnet::softmaxCrossEntropy, shuffleGenerator);
        tests.expectNear(epochLoss, nnet::evaluateLoss(*model, inputs, targets, nnet::softmaxCrossEntropy),
            1.0e-12, "the epoch loss counts every row once and weights a short last batch by its size");
    }

    // ---- the row order comes from the caller's generator -------------------
    {
        auto first = buildModel(2, 3, 2, 21);
        auto second = buildModel(2, 3, 2, 21);
        auto other = buildModel(2, 3, 2, 21);
        std::mt19937 firstGenerator(4);
        std::mt19937 secondGenerator(4);
        std::mt19937 otherGenerator(5);
        nnet::trainEpoch(*first, inputs, targets, 1, 0.3, nnet::softmaxCrossEntropy, firstGenerator);
        nnet::trainEpoch(*second, inputs, targets, 1, 0.3, nnet::softmaxCrossEntropy, secondGenerator);
        nnet::trainEpoch(*other, inputs, targets, 1, 0.3, nnet::softmaxCrossEntropy, otherGenerator);
        tests.expectTrue(sameParameters(*first, *second, 0.0),
            "the same shuffle seed gives exactly the same training");
        tests.expectTrue(!sameParameters(*first, *other, 1.0e-12),
            "a different shuffle seed visits the rows in a different order");
    }

    // ---- accuracy and evaluateLoss -----------------------------------------
    {
        // A Dense layer with an identity weight and its zero starting bias
        // passes its input straight through, so the predictions are the inputs.
        std::mt19937 generator(1);
        nnet::Dense passThrough(2, 2, generator);
        passThrough.weight.value = nnet::Tensor({2, 2}, {1, 0, 0, 1});
        const nnet::Tensor scores({3, 2}, {0.9, 0.1, 0.2, 0.8, 0.6, 0.4});
        const nnet::Tensor classes({3, 2}, {1, 0, 0, 1, 0, 1});
        tests.expectNear(nnet::accuracy(passThrough, scores, classes), 2.0 / 3.0, 1.0e-12,
            "accuracy takes the highest score in each row as the guess");

        nnet::Dense passThroughOne(1, 1, generator);
        passThroughOne.weight.value = nnet::Tensor({1, 1}, {1});
        const nnet::Tensor probabilities({3, 1}, {0.7, 0.2, 0.4});
        const nnet::Tensor answers({3, 1}, {1, 0, 1});
        tests.expectNear(nnet::accuracy(passThroughOne, probabilities, answers), 2.0 / 3.0, 1.0e-12,
            "with one column, accuracy reads 0.5 and above as yes");
        tests.expectNear(nnet::evaluateLoss(passThroughOne, probabilities, answers, nnet::binaryCrossEntropy),
            -(std::log(0.7) + std::log(0.8) + std::log(0.4)) / 3.0, 1.0e-12,
            "evaluateLoss works with binaryCrossEntropy as well");
    }

    // ---- bad arguments -----------------------------------------------------
    {
        auto model = buildModel(2, 3, 2, 21);
        std::mt19937 generator(4);
        bool rejectedBatch = false;
        try {
            nnet::trainEpoch(*model, inputs, targets, 0, 0.1, nnet::softmaxCrossEntropy, generator);
        } catch (const std::invalid_argument&) {
            rejectedBatch = true;
        }
        tests.expectTrue(rejectedBatch, "trainEpoch rejects a batch size of zero");

        bool rejectedRows = false;
        try {
            nnet::trainEpoch(*model, inputs, nnet::Tensor({4, 2}, {1, 0, 0, 1, 1, 0, 0, 1}),
                2, 0.1, nnet::softmaxCrossEntropy, generator);
        } catch (const std::invalid_argument&) {
            rejectedRows = true;
        }
        tests.expectTrue(rejectedRows, "trainEpoch rejects inputs and targets with different row counts");
    }

    // ---- it actually learns ------------------------------------------------
    // Thirty points in three clusters around three centres. A small network
    // trained in batches of 5 should separate them completely.
    {
        std::mt19937 dataGenerator(5);
        std::uniform_real_distribution<double> offset(-0.5, 0.5);
        const double centres[3][2] = {{2.0, 0.0}, {-1.0, 1.7}, {-1.0, -1.7}};
        std::vector<double> pointValues;
        std::vector<double> clusterValues;
        for (int point = 0; point < 30; ++point) {
            const int cluster = point % 3;
            pointValues.push_back(centres[cluster][0] + offset(dataGenerator));
            pointValues.push_back(centres[cluster][1] + offset(dataGenerator));
            for (int column = 0; column < 3; ++column) {
                clusterValues.push_back(column == cluster ? 1.0 : 0.0);
            }
        }
        const nnet::Tensor points({30, 2}, pointValues);
        const nnet::Tensor clusters({30, 3}, clusterValues);

        auto model = buildModel(2, 8, 3, 3);
        std::mt19937 shuffleGenerator(4);
        const double startingLoss = nnet::evaluateLoss(*model, points, clusters, nnet::softmaxCrossEntropy);
        for (int epoch = 0; epoch < 30; ++epoch) {
            nnet::trainEpoch(*model, points, clusters, 5, 0.1, nnet::softmaxCrossEntropy, shuffleGenerator);
        }
        const double finalLoss = nnet::evaluateLoss(*model, points, clusters, nnet::softmaxCrossEntropy);
        tests.expectTrue(finalLoss < 0.1 * startingLoss,
            "training on three clusters cuts the loss by more than ten times");
        tests.expectNear(nnet::accuracy(*model, points, clusters), 1.0, 1.0e-12,
            "the trained network classifies every training point correctly");
    }
}
