#include <iostream>
#include <cmath>
#include <vector>
#include "dataframe.h"

// Create a data type to represent a layer of neurons
struct Layer {
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
};

// KIND OF ERROR BUT I NEED TO MAKE IT OUTPUT THE OUTPUT FOR ONE ROW OF THE CSV NOT THE ENTIRE CSV FILE
double netSummation(const std::vector<double>& inputs, const Layer& layer, int weightIndex) {
    double output = 0.0;
    for (int i = 0; i < inputs.size(); i++) {
        output += inputs[i] * layer.weights[weightIndex][i];
    }
    return output + layer.biases[weightIndex];
}

Layer createLayer(int inputs, int nodes) {
    Layer layer;

    layer.weights.resize(nodes, std::vector<double>(inputs));
    layer.biases.resize(nodes, 0.0);

    for (auto& nodeWeights: layer.weights) {
        for (auto& weight: nodeWeights) {
            weight = ((double)rand() / RAND_MAX) - 0.5;
        }
        
    }
    return layer;
}

double sigmoid(double x) {
    x = 1.0/(1.0 + std::exp(-x));
    return x;
}

// Create a vector to store all layers of the network representing the nnet.

std::vector<Layer> createNetwork(std::vector<std::size_t> layerSizes) {
    std::vector<Layer> network;
    network.reserve(layerSizes.size());
    for (int i = 1; i < layerSizes.size(); i++) {
         network.push_back(createLayer(layerSizes[i - 1], layerSizes[i]));
        }
        return network;
    }


// Basically take in the inputs from previous layer then combine them with weights/biases and activate them with sigmoid then return
// the mapped inputs as outputs to the next layer.

std::vector<double> feedForward(const std::vector<double>& inputs, const Layer& layer) {
    std::vector<double> output;
    output.reserve(layer.weights.size());

    for (std::size_t node = 0; node < layer.weights.size(); node++) {
        double weightedSum = netSummation(inputs, layer, node);
        output.push_back(sigmoid(weightedSum));
    }
    return output;
}

std::vector<double> forwardNetwork(const std::vector<double>& inputs, const std::vector<Layer> network) {
    std::vector<double> currentOutputs = inputs;
    for (const Layer& layer: network) {
         currentOutputs = feedForward(currentOutputs, layer);
    }
    return currentOutputs;
}


int main() { 
    const std::vector<std::size_t> layerSizes = {3, 2, 1};
    std::vector<Layer> network;
    
     //csv file turned to matrix of vectors of doubles
    std::vector<std::vector<double>> dataFrame = dataframe("data.csv");

    // finish returning a dataframe not just a vector for testing with one output node
    std::vector<std::vector<double>> predictions;
    predictions.reserve(dataFrame.size());
    //
    //create a target vector
    std::vector<double> targets;
    targets.reserve(dataFrame.size());
    for (int i = 0; i < dataFrame.size(); i++) {
        targets.push_back(dataFrame[i].back());
        dataFrame[i].pop_back();
    }
    
    network = createNetwork(layerSizes);
    for (const auto& row: dataFrame) {
        predictions.push_back(forwardNetwork(row, network));
    }
  return 0;
}

