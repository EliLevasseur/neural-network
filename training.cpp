#include "training.h"
#include <cmath>
#include <iostream>

// Make a constructor function to create trainer objects.
Trainer::Trainer(const std::vector<std::vector<double>>& predictions, const std::vector<double>& targets) {
	this->predictions = predictions;
	this->targets = targets;
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

void Trainer::getDeltas() {
      	deltas.clear();
      	deltas.reserve(targets.size());
      
	for (size_t i = 0; i < targets.size(); i++) {
      		deltas.push_back(predictions[i][0] - targets[i]);
	}
	return ;
}

