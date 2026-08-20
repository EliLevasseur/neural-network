# MLP Neural Network From Scratch

This project is my attempt at building a multilayer perceptron neural network from scratch in C++.

## Current State

### Data Processing

The project has a `DataFrame` class defined in [dataframe.h](include/dataframe.h) and [dataframe.cpp](src/dataframe.cpp). It reads a numeric CSV file into a 2D `std::vector`, where each inner vector represents one row of the dataset.

For the dataset I am currently testing with, the last value in each row is the target. `getTargets()` separates those target values from the predictor values used as the network's inputs.

### Network

The network is represented as a `std::vector<Layer>`, where each element is one layer of the network.

Each `Layer` stores:

1. **Weights**
   - Stored in a `std::vector<std::vector<double>>`.
   - Each inner vector contains all the weights entering one node.
   - A weight can be accessed with `weights[node][input]`.

2. **Biases**
   - Stored in a `std::vector<double>`.
   - Each node in the layer has one bias.

The network does not define every node as its own object. Nodes and their connections are represented through vector indexes, and a collection of functions handles the calculations between layers.

The functions in [network.cpp](src/network.cpp) currently create layers and networks, calculate weighted sums, apply the sigmoid activation function, and run inputs through the complete network. A forward pass can also save the original inputs and the output of every layer as activations, which are needed later during backpropagation.

### Training

I am currently working on the `Trainer` class in [training.cpp](src/training.cpp). It calculates one delta per node and one gradient per weight using the activations saved during the forward pass.

The next goal is to use those gradients to update the weights and biases after each row. This will make the training process stochastic gradient descent: process one row, calculate its gradients, update the network immediately, and then continue to the next row.

For now, the network is being built for binary classification with a sigmoid output and binary cross-entropy loss. The structure is still intended to stay general enough to support different layer sizes.
