#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "nnet/core/tensor_ops.h"
#include "nnet/data/mnist.h"
#include "nnet/nn/dense.h"
#include "nnet/nn/module.h"
#include "nnet/nn/sequential.h"
#include "nnet/train/trainer.h"

// Handwritten digit recognition on MNIST: 60,000 training images and 10,000
// test images the network never trains on. Run `make mnist-data` once to
// download the data, then `make mnist`.

const std::string mnistFolder = "data/mnist/";
const std::size_t pixelsPerImage = 784;
const std::size_t hiddenUnits = 128;
const std::size_t digitCount = 10;
// held back from the 60,000 training images; the test images are only scored at the end
const std::size_t validationImages = 10000;
const int epochs = 10;
const std::size_t batchSize = 64;
const double learningRate = 0.1;
const unsigned randomSeed = 42;

// The column with the highest number in one row: the network's guess for a
// row of scores, or the correct digit for a one-hot label row.
std::size_t highestColumn(const nnet::Tensor& rows, std::size_t row) {
    const std::size_t width = rows.shape()[1];
    const auto rowBegin = rows.getData().begin() + row * width;
    return static_cast<std::size_t>(std::max_element(rowBegin, rowBegin + width) - rowBegin);
}

std::string percent(double fraction) {
    char text[16];
    std::snprintf(text, sizeof(text), "%.2f%%", 100.0 * fraction);
    return text;
}

// A 28x28 image as 14 lines of 28 characters. Terminal characters are about
// twice as tall as they are wide, so each line covers two pixel rows.
std::vector<std::string> drawDigit(const nnet::Tensor& images, std::size_t row) {
    const std::string shades = " .:-=+*#%@";
    std::vector<std::string> lines;
    for (std::size_t pixelRow = 0; pixelRow < 28; pixelRow += 2) {
        std::string line;
        for (std::size_t column = 0; column < 28; column++) {
            const double top = images.at({row, pixelRow * 28 + column});
            const double bottom = images.at({row, (pixelRow + 1) * 28 + column});
            const double brightness = (top + bottom) / 2.0;
            line += shades[std::min<std::size_t>(shades.size() - 1,
                static_cast<std::size_t>(brightness * static_cast<double>(shades.size())))];
        }
        lines.push_back(line);
    }
    return lines;
}

// Draws the listed test images side by side, each with the network's guess,
// how sure it was, and the correct answer underneath.
void showDigits(const nnet::Tensor& images, const nnet::Tensor& labels,
                const nnet::Tensor& probabilities, const std::vector<std::size_t>& rows) {
    std::vector<std::vector<std::string>> drawings;
    for (const std::size_t row : rows) {
        drawings.push_back(drawDigit(images, row));
    }
    for (std::size_t line = 0; line < drawings.front().size(); line++) {
        for (const auto& drawing : drawings) {
            std::cout << "  " << drawing[line] << "  ";
        }
        std::cout << '\n';
    }
    for (const std::size_t row : rows) {
        const std::size_t guess = highestColumn(probabilities, row);
        char caption[40];
        std::snprintf(caption, sizeof(caption), "guess %zu (%5.1f%%) true %zu",
            guess, 100.0 * probabilities.at({row, guess}), highestColumn(labels, row));
        std::printf("  %-28s  ", caption);
    }
    std::cout << "\n\n";
}

nnet::MnistData loadOrExplain(const std::string& imageFile, const std::string& labelFile) {
    try {
        return nnet::loadMnist(mnistFolder + imageFile, mnistFolder + labelFile);
    } catch (const std::exception& error) {
        std::cerr << "Could not load MNIST: " << error.what() << '\n'
                  << "Run `make mnist-data` from the repository root to download it.\n";
        std::exit(1);
    }
}

int main() {
    const nnet::MnistData mnistTrain = loadOrExplain("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
    const nnet::MnistData test = loadOrExplain("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");

    // Dense, ReLU, Dense. The last layer has no activation: the loss applies
    // softmax to its 10 scores itself.
    std::mt19937 weightGenerator(randomSeed);
    std::vector<std::unique_ptr<nnet::Unary_Module>> components;
    components.push_back(std::make_unique<nnet::Dense>(pixelsPerImage, hiddenUnits, weightGenerator));
    components.push_back(std::make_unique<nnet::ReLU>());
    components.push_back(std::make_unique<nnet::Dense>(hiddenUnits, digitCount, weightGenerator));
    nnet::Sequential model(std::move(components));

    std::mt19937 validationGenerator(randomSeed);
    const nnet::TrainValidationSplit train = nnet::splitOffValidation(
        mnistTrain.images, mnistTrain.labels, validationImages, validationGenerator);

    std::cout << "Training a " << pixelsPerImage << " -> " << hiddenUnits << " -> " << digitCount
              << " network on " << train.trainInputs.shape()[0] << " MNIST images"
              << " (batches of " << batchSize << ", learning rate " << learningRate << ")\n"
              << "Before training: validation accuracy "
              << percent(nnet::accuracy(model, train.validationInputs, train.validationTargets)) << " (guessing)\n";

    std::mt19937 shuffleGenerator(randomSeed);
    nnet::SGDOptimizer optimizer(model.parameters(), learningRate);
    for (int epoch = 1; epoch <= epochs; epoch++) {
        const auto start = std::chrono::steady_clock::now();
        const double trainingLoss = nnet::trainEpoch(model, train.trainInputs, train.trainTargets,
            batchSize, optimizer, nnet::softmaxCrossEntropy, shuffleGenerator);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::printf("Epoch %2d | training loss %.4f | validation accuracy %s | %.1f s\n", epoch, trainingLoss,
            percent(nnet::accuracy(model, train.validationInputs, train.validationTargets)).c_str(), seconds);
    }

    // One pass over the test set with no graph recorded, turned into
    // probabilities so each guess can show how sure the network was.
    nnet::NoGrad noGrad;
    const nnet::Tensor probabilities = nnet::softmax(model.forward(nnet::makeLeaf(test.images))->data);
    const std::size_t testCount = test.images.shape()[0];
    std::vector<std::size_t> mistakes;
    for (std::size_t row = 0; row < testCount; row++) {
        if (highestColumn(probabilities, row) != highestColumn(test.labels, row)) {
            mistakes.push_back(row);
        }
    }
    std::cout << "\nFinal test accuracy: " << percent(1.0 - static_cast<double>(mistakes.size()) / testCount)
              << " (" << testCount - mistakes.size() << " of " << testCount
              << " images it never trained on)\n\n";

    std::cout << "The first few test images:\n";
    showDigits(test.images, test.labels, probabilities, {0, 1, 2, 3});
    std::cout << "Some it gets wrong:\n";
    mistakes.resize(std::min<std::size_t>(mistakes.size(), 4));
    if (!mistakes.empty()) {
        showDigits(test.images, test.labels, probabilities, mistakes);
    }
    return 0;
}
