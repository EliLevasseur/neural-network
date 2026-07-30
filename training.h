#include <vector>
#include <cmath>
#include <iostream>


class Trainer {

	private:
		std::vector<std::vector<double>> predictions;
		std::vector<double> targets;
		std::vector<double> loss{};
		std::vector<double> deltas{};
		

	public:
		Trainer(const std::vector<std::vector<double>>& predictions, const std::vector<double>& targets);
		void binaryCrossEntropy();
		void averageLoss();
		void getDeltas();
	};

