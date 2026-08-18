#include <vector>
#include <cmath>
#include <iostream>


class Trainer {

	private:
		std::vector<std::vector<double>> predictions;
		std::vector<double> targets;
		std::vector<double> deltas;
		std::vector<std::vector<std::vector<double>>>& weightGradients;
		int learningRate;
		

	public:
		Trainer(
			const std::vector<std::vector<double>>& predictions,
			const std::vector<double>& targets,
			const int learningRate,
			std::vector<Layer>& network,
			std::vector<std::vector<double>>& activations
			);
		void getDeltas();
		void getWeightGradients();
	};

