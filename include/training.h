#ifndef TRAINING_H
#define TRAINING_H

#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <iomanip>
#include "../include/network.h"





class Trainer {

	private:
		Network& Model;
		std::vector<Layer>& network;
		std::vector<std::vector<double>> activations;
		
		std::vector<std::vector<double>> deltas;
		std::vector<std::vector<std::vector<double>>> weightGradients;


		double learningRate;
		

	public:
		std::vector<double> predictions;

		Trainer(double learningrate, Network& Model);
		void getDeltas(double target);
		void getWeightGradients();
		void sgdOptimizer();
		void trainNetwork(const std::vector<std::vector<double>>& predictors, const std::vector<double>& targets);
		void printFitProgress(std::size_t currentEpoch, std::size_t totalEpochs, std::chrono::steady_clock::time_point startTime);
        	void fit(const std::size_t epochs, const std::vector<std::vector<double>>& predictors, const std::vector<double>& targets);	
	};

#endif
