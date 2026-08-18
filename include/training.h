#include <vector>
#include <cmath>
#include <iostream>
#include "../include/network.h"


class Trainer {

	private:
		std::vector<Layer>& network;
		std::vector<std::vector<double>> activations;

		std::vector<std::vector<double>> deltas;
		std::vector<std::vector<std::vector<double>>> weightGradients;
		

	public:
		Trainer(
			std::vector<Layer>& network,
			std::vector<std::vector<double>>& activations
			);
		void getDeltas(double& target);
		void getWeightGradients();
	};

