#include <iostream>
#include <cmath>
#include <vector>
#include "include/dataframe.h"
#include "include/network.h"
#include "include/training.h"
#include "tests/test.cpp"


int main() {
    // CONVERT CSV TO MATRIX + SPLIT TARGETS FROM PREDICTORS
    DataFrame dataFrame("data/binary_test.csv"); 
    const auto& targets = dataFrame.getTargets();
    const auto& predictors =  dataFrame.getPredictors();

    // CREATE NETWORK WITH GIVEN LAYER SIZES (NODES PER LAYER)
    const std::vector<std::size_t> layerSizes = {3, 2, 4, 1};
    Network network(layerSizes); 
    // RUN TESTS
    dfTest(dataFrame);
    return 0;
}
