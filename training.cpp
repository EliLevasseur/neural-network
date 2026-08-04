#include "training.h"
#include "network.h"

// Make a constructor function to create trainer objects.
Trainer::Trainer(
	const std::vector<std::vector<double>>& predictions,
	const std::vector<double>& targets,
	const int learningRate,
	std::vector<Layer> network
	std::vector<std::vector<double>> activations;
) {
	this->predictions = predictions;
	this->targets = targets;
	this->learningRate = learningRate;
	this->network = network;
	this->activations = activations;
}


// Because I am testing the network with binary targets for now we calculate loss using cross entropy.
void Trainer::binaryCrossEntropy() {
	loss.clear();
	loss.reserve(targets.size());


	for (size_t i = 0; i < predictions.size(); i++) {
		loss.push_back(-(targets[i] * std::log(predictions[i][0]) + (1 - targets[i]) * std::log(1 - predictions[i][0])));
	}
	return ;
}

void Trainer::averageLoss() {
	double avg = 0;
	for (auto i: loss) {avg += i;}
	avg /= loss.size();
	std::cout << avg << std::endl;
	return ;
}

void Trainer::getDeltas(double target) {
      	deltas.clear();
      	deltas.resize(network.size());
	double outputDelta = activations.back().back() - target;
	deltas.back().push_back(outputDelta);
      
	for (int layer = network.size() - 2; layer >= 0; layer--) {
      		for (std::size_t node = 0; node < network[layer].weights.size(); node++) {
			double x = 0;
			if (layer == network.size() - 2) {
				x = outputDelta * network[layer + 1].weights[0][node];			}
		 	else {
				for (std::size_t weight = 0; weight < network[layer + 1].weights.size(); weight++) {
					x += network[layer + 1].weights[weight][node] * deltas[layer + 1][weight];
					}
			}
			deltas[layer].push_back(activations[layer + 1][node]* (1 - activations[layer + 1][node]) * x);
		}
	}
	return ;
}

void Trainer::getWeightGradients() {
	weightGradients.clear();
	weightGradients.resize(network.size());
	for (size_t x = 0; x < network.size(); ++x) {
		weightGradients[x].resize(network[x].weights.size());
      		for (size_t i = 0; i < network[x].weights.size(); ++i) {
	     		weightGradients[x][i].resize(network[x].weights[i].size());
      			for (size_t j = 0; j < network[x].weights[i].size(); ++j) {
      				weightGradients[x][i][j] = (deltas[x][i]* activations[x][j]);
      				}
			}
		}
	}


