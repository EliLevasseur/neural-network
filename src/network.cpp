#include "../include/network.h"

// ================== CONSTRUCTOR ======================
Network::Network(std::vector<std::size_t> layerSizes) {
    // Create a vector to store all layers of the network representing the nnet.
    network.reserve(layerSizes.size());

    for (int i = 1; i < layerSizes.size(); i++) {
         network.push_back(createLayer(layerSizes[i - 1], layerSizes[i]));
        }
 }

// ===================== GETTERS =========================
std::vector<std::vector<double>>& Network::getActivations(const std::vector<double>& inputs){
    forwardActivations(inputs);
    return activations;
    }

std::vector<Layer>& Network::getNetwork(){
    return network;
    }

double Network::netSummation(const std::vector<double>& inputs, const Layer& layer, int weightIndex) {
    // Return the summation of the input vector with the weights/biases
    double output = 0.0;

    for (int i = 0; i < inputs.size(); i++) {
        output += inputs[i] * layer.weights[weightIndex][i];
    }
    return output + layer.biases[weightIndex];
}

double Network::sigmoid(double& x) {
    x = 1.0/(1.0 + std::exp(-x));
    return x;
}

Layer Network::createLayer(int inputs, int nodes) {
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


std::vector<double> Network::feedForward(const std::vector<double>& inputs, const Layer& layer) {
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

// RETURNS FINAL PREDICTIONS AFTER FITTING MODEL
std::vector<double> Network::forwardNetwork(const std::vector<double>& inputs) {
    // Connect the neural network together by applying all activation functions and combination functions
    // of each layer to the original input row to get a prediction matrix.
    std::vector<double> currentOutputs = inputs;

    for (const Layer& layer: network) {
         currentOutputs = feedForward(currentOutputs, layer);
    }
    return currentOutputs;
}

// FOR TRAINING MODEL RETURNS 2D MATRIX OF ALL NODES OUTPUTS
void Network::forwardActivations(const std::vector<double>& inputs) {
    activations.clear();
    std::vector<double> currentOutputs = inputs;
    activations.reserve(network.size() + 1);
    activations.push_back(inputs);

    for (const Layer& layer: network) {
         currentOutputs = feedForward(currentOutputs, layer);
         activations.push_back(currentOutputs);
    }
}

std::vector<double> Network::predict(const std::vector<std::vector<double>>& inputs) {
    std::vector<double> predictions;

    for (const auto& row : inputs) {
         predictions.push_back(forwardNetwork(row)[0]);
    }
    return predictions;
}
