#include "../include/dataframe.h"
#include <string_view>
#include <iostream>
void printFlag(std::string_view str) {
	std::cout << "\n===================== " << str << " ===================== \n";
}
	

void dfTest(DataFrame& dfObject) {
	printFlag("DATAFRAME TEST");
	printFlag("");
	printFlag("Predictors");
	dfObject.printDataFrame();
	printFlag("Targets");
	dfObject.printDataFrame(dfObject.getTargets());

}
		
