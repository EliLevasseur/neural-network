#include "../include/dataframe.h"
#include "../include/training.h"
#include <iomanip>
#include <string_view>
#include <iostream>
#include <vector>
void printFlag(std::string_view str) {
	std::cout << "\n===================== " << str << " ===================== \n";
}
	

void dfTest(DataFrame& dfObject) {
	printFlag("DATAFRAME TEST");
	printFlag("              ");
	printFlag("  Predictors  ");
	dfObject.printDataFrame();
	printFlag("Targets");
	dfObject.printDataFrame(dfObject.getTargets());

}

void displayPredictions(
	const std::vector<double>& predictions,
	const std::vector<double>& targets
) {
	printFlag("FINAL PREDICTIONS");

	if (predictions.empty()) {
		std::cout << "No predictions to display.\n";
		return;
	}
	if (predictions.size() != targets.size()) {
		std::cout << "Prediction and target counts do not match.\n";
		return;
	}

	std::cout << "+------+-------------+---------+\n"
	          << "| Row  | Probability | Result  |\n"
	          << "+------+-------------+---------+\n";

	for (std::size_t row = 0; row < predictions.size(); ++row) {
		double probability = predictions[row];
		int predictedClass = probability >= 0.5 ? 1 : 0;
		int targetClass = targets[row] >= 0.5 ? 1 : 0;
		const char* result = predictedClass == targetClass ? "right" : "wrong";

		std::cout << "| " << std::setw(4) << row + 1
		          << " | " << std::setw(11) << std::fixed << std::setprecision(4)
		          << probability
		          << " | " << std::setw(7) << result << " |\n";
	}

	std::cout << "+------+-------------+---------+\n";
}
