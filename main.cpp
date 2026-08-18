#include <iostream>
#include <cmath>
#include <vector>
#include "include/dataframe.h"
#include "include/network.h"
#include "include/training.h"


int main() {
    // CONVERT CSV TO MATRIX + SPLIT TARGETS FROM PREDICTORS
    DataFrame dataFrame("data/binary_test.csv"); 
    std::vector<std::vector<double>>& df = dataFrame.getDataFrame();
    const std::vector<double> targets = dataFrame.getTargets();
    
    // CREATE NETWORK WITH GIVEN LAYER SIZES (NODES PER LAYER)
    const std::vector<std::size_t> layerSizes = {3, 2, 4, 1};
    std::vector<Layer> network = createNetwork(layerSizes);
   
    return 0;
}


