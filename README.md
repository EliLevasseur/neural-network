# MLP Neural Network From Scratch

This project is a from-scratch C++ implementation of a multilayer perceptron. Its purpose is to make the calculations inside a neural network understandable and testable before the project grows into a more general neural-network framework.

Version `0.1` is the reference model. It supports fully connected sigmoid layers, binary classification, backpropagation, stochastic gradient descent, evaluation on held-out rows, and deterministic numerical gradient tests.

## Current Features

- Numeric CSV loading without a header row
- Configurable target-column index
- Reproducible shuffled train/test splitting with an optional seed
- Configurable fully connected layer sizes
- Weights stored as `weights[node][input]`
- Sigmoid activations
- Feedforward prediction
- Binary cross-entropy loss
- Binary classification accuracy
- Backpropagation through hidden and output layers
- Per-sample stochastic gradient descent
- Epoch-based training progress
- Optional live training-loss visualization
- Separate automated test executable
- Deterministic forward-pass, backpropagation, numerical-gradient, and SGD tests

## Data Processing

Each CSV row contains numeric predictors and one binary target. The constructor receives the zero-based index of the target column:

```cpp
DataFrame dataFrame("data/binary_test.csv", 3);
```

For a row shaped like:

```text
predictor_0,predictor_1,predictor_2,target
```

the target index is `3`. The target does not need to be the final column; the selected value is removed from every predictor row and stored separately.

Access the complete predictor and target collections with:

```cpp
const auto& predictors = dataFrame.getPredictors();
const auto& targets = dataFrame.getTargets();
```

Create shuffled training and testing sets with:

```cpp
const auto split = dataFrame.trainTestSplit(0.8, 42);
```

The first argument is the fraction assigned to training. The seed makes the row split reproducible. The returned `splitContainer` owns:

```text
XTrain
yTrain
XTest
yTest
```

Training must use only `XTrain` and `yTrain`. Evaluate the finished model with `XTest` and `yTest`.

## Network Structure

The network stores a vector of fully connected `Layer` objects. Each layer owns a weight matrix and a bias vector.

Create a network by providing every layer size, including the input size:

```cpp
Network network({3, 5, 3, 1});
```

This topology contains three inputs, hidden layers with five and three nodes, and one binary output node.

Each node calculates:

```text
weighted sum = inputs * weights + bias
activation   = sigmoid(weighted sum)
```

The final sigmoid output is interpreted as a probability:

```text
probability >= 0.5  -> class 1
probability <  0.5  -> class 0
```

## Training

Construct `Trainer` by passing the model first and the learning rate second:

```cpp
Trainer trainer(network, 0.09);
trainer.fit(1000, split.XTrain, split.yTrain);
```

For every training row, the trainer performs a forward pass, calculates deltas and weight gradients, and then applies SGD.

For a sigmoid output with binary cross-entropy, the output delta simplifies to:

```text
output delta = prediction - target
```

Generate held-out predictions and metrics after training:

```cpp
const auto predictions = network.predict(split.XTest);

const double accuracy =
    trainer.getAccuracy(predictions, split.yTest);

const double loss =
    trainer.binaryCrossEntropy(predictions, split.yTest);
```

## Complete Example

```cpp
#include "include/dataframe.h"
#include "include/network.h"
#include "include/training.h"

int main() {
    DataFrame dataFrame("data/binary_test.csv", 3);
    const auto split = dataFrame.trainTestSplit(0.8, 42);

    Network network({3, 5, 3, 1});
    Trainer trainer(network, 0.09);

    trainer.fit(1000, split.XTrain, split.yTrain);

    const auto predictions = network.predict(split.XTest);
    const double accuracy =
        trainer.getAccuracy(predictions, split.yTest);
    const double loss =
        trainer.binaryCrossEntropy(predictions, split.yTest);

    return 0;
}
```

## Project Structure

```text
neural-network/
|-- data/
|   |-- binary_test.csv
|   `-- complex_8d_test.csv
|-- include/
|   |-- dataframe.h
|   |-- network.h
|   `-- training.h
|-- src/
|   |-- dataframe.cpp
|   |-- network.cpp
|   `-- training.cpp
|-- tests/
|   |-- network_test.cpp
|   |-- test_main.cpp
|   |-- test_utils.h
|   `-- training_test.cpp
|-- visualizations/
|   `-- graphing.py
|-- main.cpp
|-- Makefile
`-- README.md
```

## Building and Running

The C++ code requires a compiler with C++17 support and GNU Make.

Build the main executable without running it:

```bash
make
```

The executable is written to `build/nnet`.

Build if necessary and run the example:

```bash
make run
```

Build and run the separate automated test executable:

```bash
make test
```

A failing test makes the test executable return a nonzero exit status.

The program also accepts a machine-readable output mode:

```bash
./build/nnet --loss-csv
```

In this mode, training writes one `epoch,loss` record per line and suppresses the progress bar and final prediction table.

## Training-Loss Graph

The optional graph requires Python 3 and Matplotlib. Installing Matplotlib inside a virtual environment is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install matplotlib
```

Then run:

```bash
make graph
```

The Make target starts the network in `--loss-csv` mode and pipes its output to `visualizations/graphing.py`. A graphical display must be available; WSL users can use WSLg.

## Automated Verification

The current tests verify:

- Exact deterministic hidden and output activations
- Prediction shape and value
- Binary cross-entropy
- Expected analytic weight and bias gradients
- Numerical finite-difference gradients for every fixture parameter
- `computeGradients()` does not update parameters
- SGD updates every fixture weight and bias correctly

The numerical gradient check is the strongest correctness gate for the current backpropagation implementation.

## Current Limitations

Version `0.1` remains a binary MLP reference implementation. It does not yet support:

- A validation split or early stopping
- Shuffling training rows between epochs
- Mini-batch training
- Activation functions other than sigmoid
- Loss functions other than binary cross-entropy
- Optimizers other than basic SGD
- Multiple output nodes or multiclass classification
- Saving and loading model parameters
- Configurable reproducible weight initialization
- A general contiguous Tensor type
- CPU-kernel, SIMD, multithreaded, or GPU backends

## Next Architectural Milestones

The reference MLP will remain available as an independent correctness oracle while the reusable framework is developed beside it.

The planned sequence is:

1. Finish input-contract and DataFrame tests.
2. Add a contiguous `nnet::Tensor` with shape and row-major strides.
3. Add standalone tensor operations.
4. Add `Parameter` for values and gradients.
5. Introduce the polymorphic `Module` base class.
6. Rebuild the MLP from modules and compare its outputs and gradients against the reference implementation.
7. Separate losses, optimizers, datasets, and training orchestration.
8. Add new architectures and optimized execution backends only after correctness parity.
