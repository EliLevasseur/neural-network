#include "../include/dataframe.h"

nnet::Tensor DataFrame::flatten(std::vector<double> targets) const {
    if (targets.size() == 0)
        throw std::invalid_argument("vector cant be empty");
    return nnet::Tensor(nnet::Tensor::Shape{targets.size()}, targets);
}

nnet::Tensor DataFrame::flatten(std::vector<std::vector<double>> predictors) const {
    if (predictors.size() == 0 || predictors[0].size() == 0)
        throw std::invalid_argument("Matrix cannot be empty foo");

    std::size_t rows = predictors.size();
    std::size_t columns = predictors[0].size();
    
    std::vector<double> flattenedData;

    for (auto& row : predictors) {
        for (auto& val : row) {
            flattenedData.push_back(val);
        }
    }

    return nnet::Tensor(nnet::Tensor::Shape{rows, columns}, flattenedData);
}
