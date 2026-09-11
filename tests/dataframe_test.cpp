#include "dataframe.h"
#include "test_utils.h"

void runDataFrameTests(TestRunner& tests) {
    tests.section("DATAFRAME TESTS");

    DataFrame dataframe("data/df_test.csv", 2);

    splitContainer split = dataframe.trainTestSplit(0.5, 42);

    nnet::Tensor test_XTensor = dataframe.flatten(split.XTest);
    nnet::Tensor test_yTensor = dataframe.flatten(split.yTest);
    nnet::Tensor train_XTensor = dataframe.flatten(split.XTrain);
    nnet::Tensor train_yTensor = dataframe.flatten(split.yTrain);

    std::vector<std::vector<double>> test_df = {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0},
                {7.0, 8.0, 9.0}
    };
    
    std::cout << "tests_df =\n"; 
    dataframe.printDataFrame(test_df);

    nnet::Tensor tensor = dataframe.flatten(test_df);
    
    tests.expectTrue(tensor.rank() == 2, "tensor detived from test_df is two dimensions");
    tests.expectTrue(tensor.numel() == 9, "tensor derived from test_df contains 9 elements");
    tests.expectTrue(tensor.strides() == nnet::Tensor::Shape{3, 1}, "tensor derived from test_df has strides {3, 1}");



    tests.expectTrue(test_XTensor.shape() == nnet::Tensor::Shape{2, 2}, "XTest tensor shape is correct");
    tests.expectTrue(test_yTensor.shape() == nnet::Tensor::Shape{2}, "yTest tensor shape is correct");
    tests.expectTrue(train_XTensor.shape() == nnet::Tensor::Shape{2, 2}, "XTrain tensor shape is correct");
    tests.expectTrue(train_yTensor.shape() == nnet::Tensor::Shape{2}, "yTrain tensor shape is correct");

}
