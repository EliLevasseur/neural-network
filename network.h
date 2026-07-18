#ifndef NETWORK_H
#define NETWORK_H

#include <vector>

struct Layer {
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
};

Layer createLayer(int inputs, int nodes);

std::vector<Layer> createNetwork(
    std::vector<std::size_t> layerSizes
);

double netSummation(const std::vector<double>& inputs, const Layer& layer, int weightIndex);

double sigmoid(double x);

std::vector<double> feedForward(const std::vector<double>& inputs, const Layer& layer); 

std::vector<double> forwardNetwork(
    const std::vector<double>& inputs,
    const std::vector<Layer> network
);

#endif
