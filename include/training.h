#ifndef TRAINING_H
#define TRAINING_H

#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cassert>
#include <algorithm>
#include "../include/network.h"

class Trainer {

	private:
		Network& Model;
		std::vector<Layer>& network;
		const double learningRate;

		std::vector<std::vector<double>> activations;
		
		std::vector<std::vector<double>> deltas;
		std::vector<std::vector<std::vector<double>>> weightGradients;

	public:
		std::vector<double> predictions;

		// CONSTRUCTOR
		Trainer(Network& Model, double learningrate);

		// GETTERS
		const std::vector<std::vector<std::vector<double>>>& getWeightGradients() const;
		const std::vector<std::vector<double>>& getDeltas() const;

		// CALCULATIONS FOR TRAINING
		double getAccuracy(const std::vector<double>& predictions, const std::vector<double>& targets) const;
		double binaryCrossEntropy(const std::vector<double>& predictions, const std::vector<double>& targets) const;


		void calculateDeltas(double target);
		void calculateWeightGradients();
		// TRAINING SEQUENCE
		void computeGradients(const std::vector<double>& predictor, const double& target);
		void trainNetwork(const std::vector<std::vector<double>>& predictors, const std::vector<double>& targets);
		void fit(
			const std::size_t epochs,
			const std::vector<std::vector<double>>& predictors,
			const std::vector<double>& targets,
			bool outputLossCsv = false
		);

		//OPTIMIZERS
		void sgdOptimizer();

		// HELPER FUNCTIONS
		void printFitProgress(std::size_t currentEpoch, std::size_t totalEpochs, std::chrono::steady_clock::time_point startTime);
	};

#endif
