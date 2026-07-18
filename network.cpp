#include <iostream>
#include <cmath>
#include <vector>
#include "network.h"

double netSummation(const std::vector<double>& inputs, const Layer& layer, int weightIndex) {
    // Return the summation of the input vector with the weights/biases
    double output = 0.0;

    for (int i = 0; i < inputs.size(); i++) {
        output += inputs[i] * layer.weights[weightIndex][i];
    }
    return output + layer.biases[weightIndex];
}

double sigmoid(double x) {
    x = 1.0/(1.0 + std::exp(-x));
    return x;
}

Layer createLayer(int inputs, int nodes) {
    // Creates a fully connected layer with one weight per input-node connection.
    // Weights are randomized in [-0.5, 0.5], and each node’s bias starts at zero.
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

std::vector<Layer> createNetwork(std::vector<std::size_t> layerSizes) {
    // Create a vector to store all layers of the network representing the nnet.
    std::vector<Layer> network;

    network.reserve(layerSizes.size());

    for (int i = 1; i < layerSizes.size(); i++) {
         network.push_back(createLayer(layerSizes[i - 1], layerSizes[i]));
        }
    return network;
}

std::vector<double> feedForward(const std::vector<double>& inputs, const Layer& layer) {
    // Basically take in the inputs from previous layer then combine them with weights/biases and activate them with sigmoid then return
    // the mapped inputs as outputs to the next layer.
    std::vector<double> output;

    output.reserve(layer.weights.size());

    for (std::size_t node = 0; node < layer.weights.size(); node++) {
        double weightedSum = netSummation(inputs, layer, node);
        output.push_back(sigmoid(weightedSum));
    }
    return output;
}

std::vector<double> forwardNetwork(const std::vector<double>& inputs, const std::vector<Layer> network) {
    // Connect the neural network together by applying all activation functions and combination functions
    // of each layer to the original input row to get a prediction matrix.
    std::vector<double> currentOutputs = inputs;

    for (const Layer& layer: network) {
         currentOutputs = feedForward(currentOutputs, layer);
    }
    return currentOutputs;
}
