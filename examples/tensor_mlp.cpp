#include "nnet/core/tensor.h"
#include "nnet/core/tensor_ops.h"
#include "dataframe.h"

const std::vector<size_t> networkShape = {8, 6, 4, 1};



int main() {

    nnet::Tensor df = DataFrame("data/complex_8d_test.csv", 8).flatten();
    
    
}