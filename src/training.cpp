#include "../include/training.h"

Trainer::Trainer(
	double learningRate,
	Network& Model
	) : learningRate(learningRate), Model(Model), network(Model.getNetwork()) {}



// ================== OPTIMIZERS =======================
void Trainer::sgdOptimizer() {
	for (std::size_t layer = 0; layer < network.size(); layer++) {
      		for (std::size_t node = 0; node < network[layer].weights.size(); node++) {
      			for (std::size_t input = 0; input < network[layer].weights[node].size(); input++) {
      				network[layer].weights[node][input] -= learningRate * weightGradients[layer][node][input];
      			}
			network[layer].biases[node] -= learningRate * deltas[layer][node];
		}
	}
}

// ================== TRAINING SEQUENCE =======================

void Trainer::getDeltas(double target) {
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
	for (std::size_t layer = 0; layer < network.size(); layer++) {
		weightGradients[layer].resize(network[layer].weights.size());
      		for (std::size_t node = 0; node < network[layer].weights.size(); node++) {
	     		weightGradients[layer][node].resize(network[layer].weights[node].size());
      			for (std::size_t input = 0; input < network[layer].weights[node].size(); input++) {
      				weightGradients[layer][node][input] = (deltas[layer][node] * activations[layer][input]);
      			}
		}
	}
}

void Trainer::trainNetwork(const std::vector<std::vector<double>>& predictors, const std::vector<double>& targets) {
	for (std::size_t row = 0; row < targets.size(); row++) {
		activations.clear();
		activations = Model.getActivations(predictors[row]);
      	getDeltas(targets[row]);
		getWeightGradients();
		sgdOptimizer();
	}
}

void Trainer::fit(const std::size_t epochs, const std::vector<std::vector<double>>& predictors, const std::vector<double>& targets) {
	auto startTime = std::chrono::steady_clock::now();

	for (std::size_t epoch = 0; epoch < epochs; epoch++) {
      		trainNetwork(predictors, targets);
            printFitProgress(epoch + 1, epochs, startTime);
	}
}

// ================== HELPER FUNCTIONS =======================
//
// Yes i let ai make the progress bar thats it though
void Trainer::printFitProgress(
    std::size_t currentEpoch,
    std::size_t totalEpochs,
    std::chrono::steady_clock::time_point startTime
) {
    constexpr std::size_t barWidth = 30;
    constexpr char spinner[] = {'|', '/', '-', '\\'};

    double progress =
        static_cast<double>(currentEpoch) /
        static_cast<double>(totalEpochs);

    std::size_t completed =
        static_cast<std::size_t>(progress * barWidth);

    double elapsedSeconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - startTime
        ).count();

    std::cout << '\r'
              << spinner[currentEpoch % 4]
              << " [";

    for (std::size_t position = 0;
         position < barWidth;
         ++position) {

        if (position < completed) {
            std::cout << '#';
        } else {
            std::cout << '-';
        }
    }

    std::cout << "] "
              << currentEpoch
              << "/"
              << totalEpochs
              << " "
              << std::fixed
              << std::setprecision(1)
              << progress * 100.0
              << "% | "
              << std::setprecision(2)
              << elapsedSeconds
              << "s"
              << std::flush;

    if (currentEpoch == totalEpochs) {
        std::cout << '\n';
    }
}
