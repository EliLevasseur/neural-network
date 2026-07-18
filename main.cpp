#include <iostream>
#include <cmath>
#include <vector>
#include "dataframe.h"
#include "network.h"

int main() { 
    const std::vector<std::size_t> layerSizes = {3, 2, 4, 1};
    std::vector<Layer> network = createNetwork(layerSizes);
    
    // Csv file turned to matrix of vectors of doubles
    std::vector<std::vector<double>> dataFrame = dataframe("data.csv");

    // Finish returning a dataframe not just a vector for testing with one output node
    std::vector<std::vector<double>> predictions;

    predictions.reserve(dataFrame.size());

    // Create a target vector
    std::vector<double> targets;
    targets.reserve(dataFrame.size());

    for (int i = 0; i < dataFrame.size(); i++) {
        targets.push_back(dataFrame[i].back());
        dataFrame[i].pop_back();
    }

    // Run the rows of the dataFrame through the network and store predictions in vector
    for (const auto& row: dataFrame) {
        predictions.push_back(forwardNetwork(row, network));
    }

    // TEST FOR DEMO NETWORK WITH NO BACKPROPOGATION
    std::cout << "[" << std::endl;
    for (auto& x: predictions) {
        std::cout << "\t[";

        for (double j: x) {
            std::cout << j << ",";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "]" << std::endl;

    return 0;
}
