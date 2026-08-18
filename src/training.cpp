#include "../include/training.h"
#include "../include/network.h"

Trainer::Trainer(
	std::vector<Layer>& network,
	std::vector<std::vector<double>>& activations
) : network(network), activations(activations) {}

void Trainer::getDeltas(double& target) {
      	deltas.clear();
      	deltas.resize(network.size());
	double outputDelta = activations.back().back() - target;
	deltas.back().push_back(outputDelta);
      
	for (int layer = network.size() - 2; layer >= 0; layer--) {
      		for (std::size_t node = 0; node < network[layer].weights.size(); node++) {
			double x = 0;
			if (layer == network.size() - 2) {
				x = outputDelta * network[layer + 1].weights[0][node]; }
		 	else {
				for (std::size_t weight = 0; weight < network[layer + 1].weights.size(); weight++) {
					x += network[layer + 1].weights[weight][node] * deltas[layer + 1][weight];
					}
			}
			deltas[layer].push_back(activations[layer + 1][node]* (1 - activations[layer + 1][node]) * x);
		}
	}
}

void Trainer::getWeightGradients() {
	weightGradients.clear();
	weightGradients.resize(network.size());
	for (size_t layer = 0; layer < network.size(); layer++) {
		weightGradients[layer].resize(network[layer].weights.size());
      		for (size_t node = 0; node < network[layer].weights.size(); node++) {
	     		weightGradients[layer][node].resize(network[layer].weights[node].size());
      			for (size_t input = 0; input < network[layer].weights[node].size(); input++) {
      				weightGradients[layer][node][input] = (deltas[layer][node] * activations[layer][input]);
      				}
			}
		}
	}


