# MLP Neural Network From Scratch

This project is my implementation of a multilayer perceptron neural network built from scratch in C++. The goal is to understand what is actually happening inside a neural network instead of relying on an existing machine-learning library.

Version `0.1` can read a numeric CSV dataset, build a configurable fully connected network, train it using backpropagation and stochastic gradient descent, and make binary predictions.

## Current Features

- Numeric CSV file loading
- Automatic separation of predictors and targets
- Configurable network layer sizes
- Fully connected layers with weights and biases
- Sigmoid activation
- Complete feedforward pass
- Backpropagation through hidden and output layers
- Stochastic gradient descent
- Epoch-based model fitting
- Terminal training progress
- Binary prediction output

## Data Processing

The `DataFrame` class reads a numeric CSV file and stores it as a two-dimensional `std::vector`.

Each row represents one sample:

```text
predictor_1,predictor_2,predictor_3,target
```

For example:

```text
0.20,0.70,0.70,1
0.10,0.20,0.30,0
```

The final value in each row is treated as the target. After loading the file, the class separates the data into:

- A two-dimensional predictor matrix
- A one-dimensional target vector

These can be accessed using:

```cpp
const auto& predictors = dataFrame.getPredictors();
const auto& targets = dataFrame.getTargets();
```

The current CSV reader expects numeric data without a header row.

## Network Structure

The network is represented by the `Network` class. Internally, it stores a vector of `Layer` objects.

Each layer contains:

- A two-dimensional weight matrix
- A one-dimensional bias vector

Weights are stored using the following layout:

```cpp
weights[node][input]
```

This means every node stores one weight for every value entering it from the previous layer.

A network is created by passing a vector containing the number of nodes in each layer:

```cpp
const std::vector<std::size_t> layerSizes = {3, 5, 3, 1};
Network network(layerSizes);
```

In this example:

- The input contains 3 predictor values
- The first hidden layer contains 5 nodes
- The second hidden layer contains 3 nodes
- The output layer contains 1 node

The first value only describes the input size. Every value after it creates a fully connected layer.

## Feedforward

During a forward pass, each node calculates a weighted sum:

```text
weighted sum = inputs * weights + bias
```

The sigmoid activation function is then applied:

```text
sigmoid(x) = 1 / (1 + e^(-x))
```

The output of one layer becomes the input to the next layer. The final output is treated as the prediction probability.

For binary classification:

```text
probability >= 0.5  -> class 1
probability < 0.5   -> class 0
```

During training, the network also saves the original input and every layer's activations because they are needed during backpropagation.

## Training

Training is handled by the `Trainer` class.

For every row in the dataset, the trainer:

1. Runs the row through the network
2. Saves the activations from the forward pass
3. Calculates the output delta
4. Propagates deltas backward through the hidden layers
5. Calculates one gradient for every weight
6. Updates every weight and bias

The update rule is:

```text
new weight = old weight - learning rate * weight gradient
```

The network updates immediately after processing each row, making the current training method stochastic gradient descent.

The output delta uses the simplified derivative produced by combining a sigmoid output with binary cross-entropy:

```text
output delta = prediction - target
```

The model can be trained for multiple epochs:

```cpp
Trainer trainer(learningRate, network);
trainer.fit(epochs, predictors, targets);
```

One epoch means the trainer has processed every row in the training dataset once.

## Making Predictions

After training, predictions can be generated with:

```cpp
const auto predictions = network.predict(predictors);
```

Each returned value is a probability between `0` and `1`.

## Example

```cpp
#include "include/dataframe.h"
#include "include/network.h"
#include "include/training.h"

int main() {
    DataFrame dataFrame("data/binary_test.csv");

    const auto& predictors = dataFrame.getPredictors();
    const auto& targets = dataFrame.getTargets();

    Network network({3, 5, 3, 1});
    Trainer trainer(0.09, network);

    trainer.fit(1000, predictors, targets);

    const auto predictions = network.predict(predictors);

    return 0;
}
```

## Project Structure

```text
neural-network/
├── data/
│   └── binary_test.csv
├── include/
│   ├── dataframe.h
│   ├── network.h
│   └── training.h
├── src/
│   ├── dataframe.cpp
│   ├── network.cpp
│   └── training.cpp
├── tests/
│   └── test.cpp
├── main.cpp
├── Makefile
└── README.md
```

## Building and Running

The project includes a Makefile that compiles the source files, adds an optional flag to run tests
```bash
make
```

Compile with test output enabled:

```bash
make test
```

The compiled executable is stored at:

```text
build/nnet
```

## Current Limitations

Version `0.1` is currently focused on binary classification.

The project does not yet support:

- Train, validation, and test splitting
- Loss and accuracy tracking
- Dataset shuffling
- Mini-batch training
- Activation functions other than sigmoid
- Multiple loss functions
- Optimizers other than basic SGD
- Multiclass classification
- Saving or loading trained models
- Reproducible random initialization

It is also not intended to compete with optimized libraries such as PyTorch or TensorFlow. The current focus is correctness, understanding the underlying calculations, and creating a base that can be expanded over time.

## Next Steps

The next major goals are:

- Add training and validation metrics
- Split datasets into training and testing data
- Shuffle samples between epochs
- Add better testing for forward and backward calculations
- Support configurable activation and loss functions
- Add additional optimizers
- Save and load network weights
- Improve the project structure as the network becomes more general