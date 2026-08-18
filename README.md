# MLP Neural Network From Scratch

## Current State

### Data Processing
The project has an internal system defined in [dataframe.h](dataframe.h) that turns a csv file into a matrix the represent the rows and columns.

### Network 
The representation of a network is a `std::vector<Layer>`, a vector of layers:
#### Layer
A Layer is a struct that stores two values:
1. **Weights**:
 - stored in a `std::vector<std::vector<double>>`
 - Has one vector of connecting weights for each node in the layer
2. **Biases**:
  - Stored in a `std::vector<double>`
  - One bias per node

The Network does not define nodes as objects, the nodes are represented through indexing and a collective of functions that map inputs to the desired output

I have created the representation of the network within [network.cpp](network.cpp) and tested a single forward pass to get the first prediction

Currently working on a class to calculate weight gradients that I will use in a SGD optimizer to calculate the updated weights for the next row of the dataset