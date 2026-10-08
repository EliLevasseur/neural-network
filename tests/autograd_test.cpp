#include "test_utils.h"
#include "nnet/core/autograd/backward.h"
#include "nnet/core/autograd/operations.h"

#include <cmath>
#include <functional>
#include <stdexcept>

// Every check here compares the gradient autograd produces against a central
// finite difference: nudge one input number up and down, and measure how far
// the result moves. If the two agree, the record's backward half is right.

namespace {

    // Sum of every number in a Tensor. backward seeds the output's gradient
    // with ones, which is exactly the gradient of this sum, so the finite
    // difference of `sumOf(f(x))` is what autograd should reproduce.
    double sumOf(const nnet::Tensor& tensor) {
        return nnet::sum(tensor);
    }

    // Checks every element of one leaf's gradient against a finite difference
    // of `forward`, which recomputes the scalar result from plain Tensors.
    void expectGradientMatches(TestRunner& tests,
                               const nnet::Value& leaf,
                               const std::function<double(const nnet::Tensor&)>& forward,
                               const char* name) {
        const double step = 1.0e-6;
        bool allMatch = true;
        for (std::size_t i = 0; i < leaf->data.numel(); ++i) {
            nnet::Tensor plus = leaf->data;
            nnet::Tensor minus = leaf->data;
            std::vector<double> plusData = plus.getData();
            std::vector<double> minusData = minus.getData();
            plusData[i] += step;
            minusData[i] -= step;
            const double numeric =
                (forward(nnet::Tensor(leaf->data.shape(), plusData)) -
                 forward(nnet::Tensor(leaf->data.shape(), minusData))) / (2.0 * step);
            const double analytic = leaf->grad.getData()[i];
            // Written this way round so a NaN counts as a mismatch: every
            // comparison with NaN is false, so `NaN > 1.0e-7` would pass.
            if (!(std::abs(numeric - analytic) <= 1.0e-7)) {
                allMatch = false;
            }
        }
        tests.expectTrue(allMatch, name);
    }
}

void runAutogradTests(TestRunner& tests) {
    tests.section("AUTOGRAD");

    // ---- step 1: sigmoid, one and two deep -------------------------------
    {
        nnet::Value x = nnet::makeLeaf(nnet::Tensor({3}, {-1, 0, 2}));
        nnet::backward(nnet::sigmoid(x));
        expectGradientMatches(tests, x,
            [](const nnet::Tensor& t) { return sumOf(nnet::sigmoid(t)); },
            "sigmoid gradient matches a finite difference");
    }
    {
        nnet::Value x = nnet::makeLeaf(nnet::Tensor({3}, {-1, 0, 2}));
        nnet::backward(nnet::sigmoid(nnet::sigmoid(x)));
        expectGradientMatches(tests, x,
            [](const nnet::Tensor& t) { return sumOf(nnet::sigmoid(nnet::sigmoid(t))); },
            "backward walks more than one operation deep");
    }
    {
        nnet::Value x = nnet::makeLeaf(nnet::Tensor({4}, {-1.5, -0.3, 0.4, 2.0}));
        nnet::backward(nnet::relu(x));
        expectGradientMatches(tests, x,
            [](const nnet::Tensor& t) { return sumOf(nnet::relu(t)); },
            "relu gradient matches a finite difference");
    }
    {
        // backward seeds the last output with ones, so a record that ignored
        // its incoming gradient would still pass the check above. A sigmoid
        // after relu makes the incoming gradient something other than one.
        nnet::Value x = nnet::makeLeaf(nnet::Tensor({4}, {-1.5, -0.3, 0.4, 2.0}));
        nnet::backward(nnet::sigmoid(nnet::relu(x)));
        expectGradientMatches(tests, x,
            [](const nnet::Tensor& t) { return sumOf(nnet::sigmoid(nnet::relu(t))); },
            "relu passes back the gradient it receives, not just its slope");
    }

    // ---- step 2: matmul, bias add, loss ----------------------------------
    {
        const nnet::Tensor left({2, 3}, {1, 2, 3, 4, 5, 6});
        const nnet::Tensor right({3, 2}, {0.1, -0.2, 0.3, 0.4, -0.5, 0.6});
        nnet::Value leftLeaf = nnet::makeLeaf(left);
        nnet::Value rightLeaf = nnet::makeLeaf(right);
        nnet::backward(nnet::matmul(leftLeaf, rightLeaf));
        expectGradientMatches(tests, leftLeaf,
            [&](const nnet::Tensor& t) { return sumOf(t * right); },
            "matmul gradient for the left input matches a finite difference");
        expectGradientMatches(tests, rightLeaf,
            [&](const nnet::Tensor& t) { return sumOf(left * t); },
            "matmul gradient for the right input matches a finite difference");
    }
    {
        const nnet::Tensor values({2, 3}, {1, 2, 3, 4, 5, 6});
        const nnet::Tensor bias({3}, {10, 20, 30});
        nnet::Value valuesLeaf = nnet::makeLeaf(values);
        nnet::Value biasLeaf = nnet::makeLeaf(bias);
        nnet::backward(nnet::addBias(valuesLeaf, biasLeaf));
        expectGradientMatches(tests, valuesLeaf,
            [&](const nnet::Tensor& t) { nnet::Tensor out = t; nnet::addBias(out, bias); return sumOf(out); },
            "bias add passes the values' gradient through unchanged");
        expectGradientMatches(tests, biasLeaf,
            [&](const nnet::Tensor& t) { nnet::Tensor out = values; nnet::addBias(out, t); return sumOf(out); },
            "bias add sums the bias gradient over every row");
        tests.expectTrue(valuesLeaf->data.getData() == values.getData(),
            "bias add leaves its input's numbers untouched");
    }
    {
        const nnet::Tensor target({2}, {1.0, 0.0});
        nnet::Value prediction = nnet::makeLeaf(nnet::Tensor({2}, {0.9, 0.2}));
        nnet::Value targetLeaf = nnet::makeLeaf(target);
        const nnet::Value loss = nnet::binaryCrossEntropy(prediction, targetLeaf);
        tests.expectTrue(loss->data.shape() == nnet::Tensor::Shape{1},
            "the loss is a single number carried as shape {1}");
        nnet::backward(loss);
        expectGradientMatches(tests, prediction,
            [&](const nnet::Tensor& t) { return nnet::binaryCrossEntropy(t, target); },
            "loss gradient for the prediction matches a finite difference");
        tests.expectTrue(targetLeaf->grad.getData() == std::vector<double>{0.0, 0.0},
            "the target receives no gradient");
    }

    // ---- all four together: one Dense layer, sigmoid, and the loss -------
    {
        const nnet::Tensor input({1, 2}, {0.6, -0.4});
        const nnet::Tensor weight({2, 1}, {-0.3, 0.2});
        const nnet::Tensor bias({1}, {0.05});
        const nnet::Tensor target({1, 1}, {1.0});

        nnet::Value weightLeaf = nnet::makeLeaf(weight);
        nnet::Value biasLeaf = nnet::makeLeaf(bias);
        const nnet::Value prediction = nnet::sigmoid(nnet::addBias(
            nnet::matmul(nnet::makeLeaf(input), weightLeaf), biasLeaf));
        nnet::backward(nnet::binaryCrossEntropy(prediction, nnet::makeLeaf(target)));

        auto lossWithWeight = [&](const nnet::Tensor& w) {
            nnet::Tensor out = input * w; nnet::addBias(out, bias);
            return nnet::binaryCrossEntropy(nnet::sigmoid(out), target);
        };
        auto lossWithBias = [&](const nnet::Tensor& b) {
            nnet::Tensor out = input * weight; nnet::addBias(out, b);
            return nnet::binaryCrossEntropy(nnet::sigmoid(out), target);
        };
        expectGradientMatches(tests, weightLeaf, lossWithWeight,
            "a full layer's weight gradient matches a finite difference");
        expectGradientMatches(tests, biasLeaf, lossWithBias,
            "a full layer's bias gradient matches a finite difference");
    }

    // ---- softmax cross-entropy: the multi-class loss ----------------------
    {
        const nnet::Tensor target({2, 3}, {1, 0, 0, 0, 0, 1});
        nnet::Value scores = nnet::makeLeaf(nnet::Tensor({2, 3}, {2.0, 1.0, 0.1, 0.5, 0.5, 3.0}));
        nnet::backward(nnet::softmaxCrossEntropy(scores, nnet::makeLeaf(target)));
        expectGradientMatches(tests, scores,
            [&](const nnet::Tensor& t) { return nnet::softmaxCrossEntropy(t, target); },
            "softmax cross-entropy gradient matches a finite difference");
    }
    {
        // Starting backward from 2.5 instead of 1 must scale every gradient
        // by 2.5, which only happens if the record uses what it receives.
        const nnet::Tensor target({2, 3}, {1, 0, 0, 0, 0, 1});
        nnet::Value scores = nnet::makeLeaf(nnet::Tensor({2, 3}, {2.0, 1.0, 0.1, 0.5, 0.5, 3.0}));
        nnet::backward(nnet::softmaxCrossEntropy(scores, nnet::makeLeaf(target)), nnet::Tensor({1}, {2.5}));
        expectGradientMatches(tests, scores,
            [&](const nnet::Tensor& t) { return 2.5 * nnet::softmaxCrossEntropy(t, target); },
            "softmax cross-entropy passes back the gradient it receives");
    }

    // ---- the MNIST network in miniature: Dense, ReLU, Dense, loss ---------
    // Two examples, three inputs, four hidden units, three classes. Every
    // hidden pre-activation is at least 0.12 away from zero, where ReLU bends.
    {
        const nnet::Tensor input({2, 3}, {0.6, -0.4, 0.9, -0.2, 0.7, 0.1});
        const nnet::Tensor hiddenWeight({3, 4}, {0.5, -0.3, 0.2, -0.6,
                                                 -0.4, 0.8, 0.1, 0.3,
                                                 0.2, 0.1, -0.7, 0.4});
        const nnet::Tensor hiddenBias({4}, {0.05, -0.1, 0.2, 0.0});
        const nnet::Tensor outputWeight({4, 3}, {0.3, -0.2, 0.5,
                                                 -0.1, 0.4, 0.2,
                                                 0.6, -0.5, 0.1,
                                                 -0.3, 0.2, -0.4});
        const nnet::Tensor outputBias({3}, {0.1, -0.05, 0.0});
        const nnet::Tensor target({2, 3}, {0, 1, 0, 0, 0, 1});

        nnet::Value hiddenWeightLeaf = nnet::makeLeaf(hiddenWeight);
        nnet::Value hiddenBiasLeaf = nnet::makeLeaf(hiddenBias);
        nnet::Value outputWeightLeaf = nnet::makeLeaf(outputWeight);
        nnet::Value outputBiasLeaf = nnet::makeLeaf(outputBias);
        const nnet::Value hiddenActivation = nnet::relu(nnet::addBias(
            nnet::matmul(nnet::makeLeaf(input), hiddenWeightLeaf), hiddenBiasLeaf));
        const nnet::Value scores = nnet::addBias(
            nnet::matmul(hiddenActivation, outputWeightLeaf), outputBiasLeaf);
        nnet::backward(nnet::softmaxCrossEntropy(scores, nnet::makeLeaf(target)));

        auto lossFrom = [&](const nnet::Tensor& firstWeight, const nnet::Tensor& firstBias,
                            const nnet::Tensor& secondWeight, const nnet::Tensor& secondBias) {
            nnet::Tensor hiddenPreActivation = input * firstWeight;
            nnet::addBias(hiddenPreActivation, firstBias);
            nnet::Tensor outputScores = nnet::relu(hiddenPreActivation) * secondWeight;
            nnet::addBias(outputScores, secondBias);
            return nnet::softmaxCrossEntropy(outputScores, target);
        };
        expectGradientMatches(tests, hiddenWeightLeaf,
            [&](const nnet::Tensor& w) { return lossFrom(w, hiddenBias, outputWeight, outputBias); },
            "hidden weight gradient through relu and softmax cross-entropy matches a finite difference");
        expectGradientMatches(tests, hiddenBiasLeaf,
            [&](const nnet::Tensor& b) { return lossFrom(hiddenWeight, b, outputWeight, outputBias); },
            "hidden bias gradient through relu and softmax cross-entropy matches a finite difference");
        expectGradientMatches(tests, outputWeightLeaf,
            [&](const nnet::Tensor& w) { return lossFrom(hiddenWeight, hiddenBias, w, outputBias); },
            "output weight gradient through softmax cross-entropy matches a finite difference");
        expectGradientMatches(tests, outputBiasLeaf,
            [&](const nnet::Tensor& b) { return lossFrom(hiddenWeight, hiddenBias, outputWeight, b); },
            "output bias gradient through softmax cross-entropy matches a finite difference");
    }
    // ---- step 3: one value feeding two places -----------------------------
    // h feeds two separate sigmoid records. Both contributions must reach h
    // before h's own record passes anything back to x. Without the "seen it"
    // set, h would be listed twice and x would get double. With the wrong
    // order, h's record would run after only one contribution and x would get
    // half.
    {
        nnet::Value x = nnet::makeLeaf(nnet::Tensor({3}, {-1, 0, 2}));
        const nnet::Value h = nnet::sigmoid(x);
        nnet::backward(nnet::add(nnet::sigmoid(h), nnet::sigmoid(h)));
        expectGradientMatches(tests, x,
            [](const nnet::Tensor& t) {
                const nnet::Tensor once = nnet::sigmoid(nnet::sigmoid(t));
                return sumOf(once + once);
            },
            "a value feeding two records collects both gradients before passing them on");
    }
    {
        // The same value in both input slots of one record.
        nnet::Value x = nnet::makeLeaf(nnet::Tensor({3}, {-1, 0, 2}));
        const nnet::Value h = nnet::sigmoid(x);
        nnet::backward(nnet::add(h, h));
        expectGradientMatches(tests, x,
            [](const nnet::Tensor& t) {
                const nnet::Tensor once = nnet::sigmoid(t);
                return sumOf(once + once);
            },
            "a value used twice by one record gets both contributions");
    }

    // ---- one backward per graph ----------------------------------------------
    {
        nnet::Tensor persistent = nnet::zeros({3});
        nnet::Value x = nnet::makeLeaf(nnet::Tensor({3}, {-1, 0, 2}));
        x->accumulateInto = &persistent;
        const nnet::Value y = nnet::sigmoid(x);
        nnet::backward(y);
        const std::vector<double> afterFirst = persistent.getData();

        tests.expectTrue(y->madeBy->consumed && y->madeBy->inputs.empty(),
            "backward marks its records as used and releases their inputs");

        bool rejected = false;
        try { nnet::backward(y); }
        catch (const std::logic_error&) { rejected = true; }
        tests.expectTrue(rejected, "a second backward on the same graph is refused");
        tests.expectTrue(persistent.getData() == afterFirst,
            "a refused backward deposits nothing a second time");

        // A fresh forward pass is a fresh graph, so it is allowed.
        nnet::backward(nnet::sigmoid(x));
        const std::vector<double> afterSecond = persistent.getData();
        bool doubled = true;
        for (std::size_t i = 0; i < afterFirst.size(); ++i) {
            doubled = doubled && std::abs(afterSecond[i] - 2.0 * afterFirst[i]) < 1.0e-12;
        }
        tests.expectTrue(doubled, "a new graph over the same leaf adds its own contribution");
    }

    // ---- backward from a supplied gradient ------------------------------------
    {
        nnet::Value x = nnet::makeLeaf(nnet::Tensor({3}, {-1, 0, 2}));
        nnet::backward(nnet::sigmoid(x), nnet::Tensor({3}, {1, 2, 3}));
        const nnet::Tensor derivative = nnet::sigmoidDerivitive(x->data);
        tests.expectNear(x->grad.at({2}), 3.0 * derivative.at({2}), 1.0e-12,
            "backward can start from a supplied gradient instead of ones");

        bool rejected = false;
        try { nnet::backward(nnet::sigmoid(x), nnet::Tensor({2}, {1, 1})); }
        catch (const std::invalid_argument&) { rejected = true; }
        tests.expectTrue(rejected, "a supplied gradient of the wrong shape is refused");
    }

    // ---- no-grad ------------------------------------------------------------
    {
        nnet::Value x = nnet::makeLeaf(nnet::Tensor({3}, {-1, 0, 2}));
        nnet::Value recorded = nnet::sigmoid(x);
        nnet::Value unrecorded;
        {
            nnet::NoGrad noGrad;
            unrecorded = nnet::sigmoid(x);
            {
                nnet::NoGrad nested;
            }
            tests.expectTrue(!nnet::gradientRecordingEnabled,
                "leaving a nested NoGrad keeps recording off for the outer one");
        }
        tests.expectTrue(unrecorded->madeBy == nullptr,
            "an operation inside NoGrad leaves no record");
        tests.expectTrue(unrecorded->data.getData() == recorded->data.getData(),
            "NoGrad changes nothing about the numbers computed");
        tests.expectTrue(nnet::gradientRecordingEnabled && nnet::sigmoid(x)->madeBy != nullptr,
            "recording comes back on when NoGrad goes out of scope");
    }

}
