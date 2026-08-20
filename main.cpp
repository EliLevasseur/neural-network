#include "include/dataframe.h"
#include "include/network.h"
#include "include/training.h"
#include "tests/test.cpp"

const double LEARNING_RATE = 0.9;
const std::size_t EPOCHS = 5000;
const std::vector<std::size_t> LAYER_SIZES = {3, 4, 1};

// Make file flag
#ifdef TEST_MODE
const bool RUN_TESTS = true;
#else
const bool RUN_TESTS = false;
#endif

int main() {
    // CONVERT CSV TO MATRIX + SPLIT TARGETS FROM PREDICTORS + TRAIN TEST SPLIT
    //
    DataFrame dataFrame("data/binary_test.csv"); 
    splitContainer split = dataFrame.trainTestSplit(0.5);

    // CREATE NETWORK WITH GIVEN LAYER SIZES (NODES PER LAYER)
    Network network(LAYER_SIZES);

    // INITIALIZE THE TRAINER BY PASSING IT THE NETWORK TO OPTIMIZE WEIGHTS
    //
    Trainer trainer(LEARNING_RATE, network);
    trainer.fit(EPOCHS, split.XTrain, split.yTrain);
    
    // GET FINAL PREDICTIONS
    //
    const auto& predictions = network.predict(split.XTest);

    // RUN TESTS
    //
    if (RUN_TESTS) {
//        dfTest(dataFrame);
        displayPredictions(predictions, split.yTest);
//        splitTest(split.XTrain, split.XTest);   
        accuracyTest(predictions, split.yTest, trainer);
        lossTest(predictions, split.yTest, trainer);
    }

    return 0;
}
