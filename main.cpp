#include "include/dataframe.h"
#include "include/network.h"
#include "include/training.h"

#include <iomanip>
#include <iostream>
#include <string_view>

const int RANDOM_SEED = 42;
const double LEARNING_RATE = 0.1;
const std::size_t EPOCHS = 1000;
const std::size_t TARGET_INDEX = 8; // This is the index of the binary target in your dataset.
const std::vector<std::size_t> LAYER_SIZES = {8, 8, 4, 1};



void displayPredictions(const std::vector<double>& predictions, const std::vector<double>& targets) {
    std::cout << "\n+------+-------------+---------+\n"
              << "| Row  | Probability | Result  |\n"
              << "+------+-------------+---------+\n";

    for (std::size_t row = 0; row < predictions.size(); ++row) {
        const double probability = predictions[row];
        const int predictedClass = probability >= 0.5 ? 1 : 0;
        const int targetClass = targets[row] >= 0.5 ? 1 : 0;
        const char* result = predictedClass == targetClass ? "right" : "wrong";

        std::cout << "| " << std::setw(4) << row + 1
                  << " | " << std::setw(11) << std::fixed << std::setprecision(4)
                  << probability
                  << " | " << std::setw(7) << result << " |\n";
    }

    std::cout << "+------+-------------+---------+\n";
}


int main(int argc, char* argv[]) {
    bool outputLossCsv = false;

    if (argc == 2 && std::string_view(argv[1]) == "--loss-csv") {
        outputLossCsv = true;
    } else if (argc != 1) {
        std::cerr << "Availble Flags: " << argv[0] << " [--loss-csv]\n";
        return 1;
    }

    // CONVERT CSV TO MATRIX + SPLIT TARGETS FROM PREDICTORS + TRAIN TEST SPLIT
    //
    DataFrame dataFrame("data/complex_8d_test.csv", TARGET_INDEX);
    splitContainer split = dataFrame.trainTestSplit(0.8, RANDOM_SEED);

    // CREATE NETWORK WITH GIVEN LAYER SIZES (NODES PER LAYER)
    Network network(LAYER_SIZES);

    // INITIALIZE THE TRAINER BY PASSING IT THE NETWORK TO OPTIMIZE WEIGHTS
    //
    Trainer trainer(network, LEARNING_RATE);
    trainer.fit(EPOCHS, split.XTrain, split.yTrain, outputLossCsv);

    // GET AND REPORT FINAL PREDICTIONS
    //
    if (!outputLossCsv) {
        const auto predictions = network.predict(split.XTest);
        displayPredictions(predictions, split.yTest);
        std::cout << "Test accuracy: " << trainer.getAccuracy(predictions, split.yTest) << '\n';
        std::cout << "Test loss: " << trainer.binaryCrossEntropy(predictions, split.yTest) << '\n';
    }

    return 0;
}
