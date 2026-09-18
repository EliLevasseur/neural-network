#include <iostream>

#include "nnet/core/tensor.h"
#include "nnet/core/tensor_ops.h"
#include "dataframe.h"

const std::vector<size_t> networkShape = {8, 6, 4, 1};



int main() {

    DataFrame data("data/complex_8d_test.csv", 8);
    nnet::Tensor df = data.flatten(data.getPredictors());
    std::vector<nnet::Paramater> weightTensors = nnet::createWeights(networkShape);

    std::vector<double> testData;
    testData.reserve(networkShape[0]);
    
    for (size_t i = 0; i < networkShape[0]; i++) {
        testData.push_back(df.at(nnet::Tensor::Shape{0, i}));
    }
    nnet::Tensor testTensor({1, networkShape[0]}, testData);
    nnet::Tensor prediction = nnet::feedForward(testTensor, weightTensors);

    std::cout << prediction.at({0, 0}) << std::endl;


    return 0;

}