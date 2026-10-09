#include "nnet/train/trainer.h"

#include "nnet/core/autograd/backward.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace nnet {

    namespace {
        void checkExamples(const Tensor& inputs, const Tensor& targets) {
            if (inputs.rank() != 2 || targets.rank() != 2) {
                throw std::invalid_argument("Inputs and targets must be [rows, columns]");
            }
            if (inputs.shape()[0] != targets.shape()[0]) {
                throw std::invalid_argument("Inputs and targets must have the same number of rows");
            }
        }
    }

    Tensor takeRows(const Tensor& data, const std::vector<std::size_t>& rowIndices) {
        const std::size_t rowCount = data.shape()[0];
        const std::size_t rowWidth = data.numel() / rowCount;
        const std::vector<double>& source = data.getData();

        std::vector<double> selected;
        selected.reserve(rowIndices.size() * rowWidth);
        for (const std::size_t row : rowIndices) {
            if (row >= rowCount) {
                throw std::out_of_range("takeRows: row index is past the last row");
            }
            const auto rowBegin = source.begin() + row * rowWidth;
            selected.insert(selected.end(), rowBegin, rowBegin + rowWidth);
        }

        Tensor::Shape shape = data.shape();
        shape[0] = rowIndices.size();
        return Tensor(shape, std::move(selected));
    }

    TrainValidationSplit splitOffValidation(const Tensor& inputs, const Tensor& targets,
                                            std::size_t validationRows, std::mt19937& generator) {
        checkExamples(inputs, targets);
        const std::size_t rowCount = inputs.shape()[0];
        if (validationRows == 0 || validationRows >= rowCount) {
            throw std::invalid_argument("Validation needs at least one row and must leave at least one training row");
        }

        std::vector<std::size_t> order(rowCount);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), generator);

        // the same index lists go to inputs and targets so every row keeps its label
        const std::vector<std::size_t> validationIndices(order.begin(), order.begin() + validationRows);
        const std::vector<std::size_t> trainIndices(order.begin() + validationRows, order.end());
        return TrainValidationSplit{takeRows(inputs, trainIndices), takeRows(targets, trainIndices),
                                    takeRows(inputs, validationIndices), takeRows(targets, validationIndices)};
    }

    double trainEpoch(Unary_Module& model, const Tensor& inputs, const Tensor& targets,
                      std::size_t batchSize, Optimizer& optimizer,
                      LossFunction loss, std::mt19937& generator) {
        checkExamples(inputs, targets);
        if (batchSize == 0) {
            throw std::invalid_argument("batchSize must be at least 1");
        }

        const std::size_t rowCount = inputs.shape()[0];
        std::vector<std::size_t> order(rowCount);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), generator);

        double weightedLossTotal = 0.0;
        for (std::size_t batchStart = 0; batchStart < rowCount; batchStart += batchSize) {
            const std::size_t batchEnd = std::min(batchStart + batchSize, rowCount);
            const std::vector<std::size_t> batchRows(order.begin() + batchStart, order.begin() + batchEnd);

            model.zeroGrad();
            const Value prediction = model.forward(makeLeaf(takeRows(inputs, batchRows)));
            const Value batchLoss = loss(prediction, makeLeaf(takeRows(targets, batchRows)));
            // weight by row count so a short last batch counts for less
            weightedLossTotal += batchLoss->data.at({0}) * static_cast<double>(batchRows.size());
            backward(batchLoss);
            optimizer.step();
        }
        return weightedLossTotal / static_cast<double>(rowCount);
    }

    double evaluateLoss(const Unary_Module& model, const Tensor& inputs, const Tensor& targets,
                        LossFunction loss) {
        checkExamples(inputs, targets);
        NoGrad noGrad;
        return loss(model.forward(makeLeaf(inputs)), makeLeaf(targets))->data.at({0});
    }

    double accuracy(const Unary_Module& model, const Tensor& inputs, const Tensor& targets) {
        checkExamples(inputs, targets);
        NoGrad noGrad;
        const Tensor predictions = model.forward(makeLeaf(inputs))->data;
        if (predictions.shape() != targets.shape()) {
            throw std::invalid_argument("The model's output must have the same shape as the targets");
        }

        const std::size_t rowCount = targets.shape()[0];
        const std::size_t width = targets.shape()[1];
        const std::vector<double>& predicted = predictions.getData();
        const std::vector<double>& expected = targets.getData();

        std::size_t correct = 0;
        for (std::size_t row = 0; row < rowCount; row++) {
            const auto predictedBegin = predicted.begin() + row * width;
            const auto expectedBegin = expected.begin() + row * width;
            bool isCorrect = false;
            if (width == 1) {
                isCorrect = (*predictedBegin >= 0.5) == (*expectedBegin >= 0.5);
            } else {
                const auto guessedClass = std::max_element(predictedBegin, predictedBegin + width) - predictedBegin;
                const auto correctClass = std::max_element(expectedBegin, expectedBegin + width) - expectedBegin;
                isCorrect = guessedClass == correctClass;
            }
            if (isCorrect) {
                ++correct;
            }
        }
        return static_cast<double>(correct) / static_cast<double>(rowCount);
    }

}
