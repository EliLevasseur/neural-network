#include <iostream>
#include <cmath>
#include <vector>
#include "include/dataframe.h"
#include "include/network.h"
#include "include/training.h"
#include "tests/test.cpp"

const double LEARNING_RATE = 0.09;
const std::size_t EPOCHS = 10000;
const std::vector<std::size_t> LAYER_SIZES = {3, 5, 3, 1};

// Make file flag
#ifdef TEST_MODE
const bool RUN_TESTS = true;
#else
const bool RUN_TESTS = false;
#endif

int main() {
    // CONVERT CSV TO MATRIX + SPLIT TARGETS FROM PREDICTORS
    //
    DataFrame dataFrame("data/binary_test.csv"); 
    const auto& targets = dataFrame.getTargets();
    const auto& predictors =  dataFrame.getPredictors();

    // CREATE NETWORK WITH GIVEN LAYER SIZES (NODES PER LAYER)
    Network network(LAYER_SIZES);

    // INITIALIZE THE TRAINER BY PASSING IT THE NETWORK TO OPTIMIZE WEIGHTS
    //
    Trainer trainer(LEARNING_RATE, network);
    trainer.fit(EPOCHS, predictors, targets);
    
    // GET FINAL PREDICTIONS
    //
    const auto& predictions = network.predict(predictors);

    // RUN TESTS
    //
    if (RUN_TESTS) {
//        dfTest(dataFrame);
        displayPredictions(predictions, targets);
        
    }

    return 0;
}
