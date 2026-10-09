CXX := g++
CXXFLAGS := -std=c++17 -g3 -O0 -Wall -Wextra -Wpedantic -Iinclude

.DEFAULT_GOAL := build

REFERENCE_SOURCE := examples/reference_mlp.cpp
REFERENCE_SOURCES := src/dataframe.cpp src/network.cpp src/training.cpp
REFERENCE_BINARY := build/reference_mlp

NETWORK_SOURCES := src/network.cpp src/training.cpp
DATAFRAME_SOURCE := src/dataframe.cpp
AUTOGRAD_SOURCES := src/autograd/operations.cpp src/autograd/backward.cpp
# The trainer runs SGD steps, so the two are always linked together.
TRAINER_SOURCES := src/train/trainer.cpp src/optim/optimizer.cpp
# Loaders that turn a dataset's own file format into input and target Tensors.
DATA_SOURCES := src/data/mnist.cpp
TENSOR_SOURCES := src/core/tensor.cpp src/core/tensor_dataframe.cpp src/core/tensor_ops.cpp $(AUTOGRAD_SOURCES) $(TRAINER_SOURCES) $(DATA_SOURCES)

REFERENCE_TEST_SOURCES := tests/reference_test_main.cpp tests/network_test.cpp tests/training_test.cpp
REFERENCE_TEST_BINARY := build/reference_tests

TENSOR_TEST_SOURCES := tests/tensor_test_main.cpp tests/tensor_test.cpp tests/dataframe_test.cpp tests/tensor_ops_tests.cpp tests/dense_test.cpp tests/autograd_test.cpp tests/trainer_test.cpp tests/mnist_test.cpp
TENSOR_TEST_BINARY := build/tensor_tests

# The parity binary is the one place the legacy oracle and the Tensor code
# are deliberately linked together, so the new gradients can be compared
# against numbers the reference model already proved. It needs no DataFrame.
PARITY_TENSOR_SOURCES := src/core/tensor.cpp src/core/tensor_ops.cpp $(AUTOGRAD_SOURCES)
PARITY_TEST_SOURCES := tests/parity_test_main.cpp tests/parity_test.cpp
PARITY_TEST_BINARY := build/parity_tests

# Example driving the new framework: Dense layers built from a shape list,
# chained by the caller with the activation applied between them.
TENSOR_MLP_SOURCE := examples/tensor_mlp.cpp
TENSOR_MLP_BINARY := build/tensor_mlp

# Handwritten digit recognition. Built with -O2 added after CXXFLAGS (the last
# -O wins), because training on 60,000 images at -O0 takes minutes per pass.
MNIST_SOURCE := examples/mnist.cpp
MNIST_BINARY := build/mnist

HEADERS := $(wildcard include/*.h include/nnet/core/*.h include/nnet/core/autograd/*.h include/nnet/nn/*.h include/nnet/train/*.h include/nnet/data/*.h include/nnet/optim/*.h)

.PHONY: build run test test-reference test-tensor test-parity tensor-mlp mnist mnist-data graph

build: $(REFERENCE_BINARY)

run: $(REFERENCE_BINARY)
	./$(REFERENCE_BINARY)

test: test-reference test-tensor test-parity

test-reference: $(REFERENCE_TEST_BINARY)
	./$(REFERENCE_TEST_BINARY)

test-tensor: $(TENSOR_TEST_BINARY)
	./$(TENSOR_TEST_BINARY)

test-parity: $(PARITY_TEST_BINARY)
	./$(PARITY_TEST_BINARY)

tensor-mlp: $(TENSOR_MLP_BINARY)
	./$(TENSOR_MLP_BINARY)

mnist: $(MNIST_BINARY)
	./$(MNIST_BINARY)

mnist-data:
	./scripts/download_mnist.sh

graph: $(REFERENCE_BINARY)
	./$(REFERENCE_BINARY) --loss-csv | python3 ./visualizations/graphing.py

$(REFERENCE_BINARY): $(REFERENCE_SOURCE) $(REFERENCE_SOURCES) $(HEADERS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(REFERENCE_SOURCE) $(REFERENCE_SOURCES) -o $(REFERENCE_BINARY)

# Reference tests link only Network/Trainer sources - no Tensor compilation
# unit is ever pulled into this binary, so a broken Tensor cannot break it.
$(REFERENCE_TEST_BINARY): $(REFERENCE_TEST_SOURCES) tests/test_utils.h $(NETWORK_SOURCES) $(HEADERS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) -Itests $(REFERENCE_TEST_SOURCES) $(NETWORK_SOURCES) -o $(REFERENCE_TEST_BINARY)

# Tensor tests link DataFrame (for the Tensor-flattening adapter) and every
# Tensor compilation unit, independently of the Network/Trainer reference path.
$(TENSOR_TEST_BINARY): $(TENSOR_TEST_SOURCES) tests/test_utils.h $(DATAFRAME_SOURCE) $(TENSOR_SOURCES) $(HEADERS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) -Itests $(TENSOR_TEST_SOURCES) $(DATAFRAME_SOURCE) $(TENSOR_SOURCES) -o $(TENSOR_TEST_BINARY)

# Parity tests intentionally link both worlds: the legacy Network/Trainer
# oracle and the Tensor operations. This is the only target that does so;
# test-reference stays free of every Tensor compilation unit.
$(PARITY_TEST_BINARY): $(PARITY_TEST_SOURCES) tests/test_utils.h $(NETWORK_SOURCES) $(PARITY_TENSOR_SOURCES) $(HEADERS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) -Itests $(PARITY_TEST_SOURCES) $(NETWORK_SOURCES) $(PARITY_TENSOR_SOURCES) -o $(PARITY_TEST_BINARY)

# The Tensor example needs DataFrame for CSV loading plus every Tensor
# compilation unit. It never touches the legacy Network or Trainer.
$(TENSOR_MLP_BINARY): $(TENSOR_MLP_SOURCE) $(DATAFRAME_SOURCE) $(TENSOR_SOURCES) $(HEADERS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(TENSOR_MLP_SOURCE) $(DATAFRAME_SOURCE) $(TENSOR_SOURCES) -o $(TENSOR_MLP_BINARY)

$(MNIST_BINARY): $(MNIST_SOURCE) $(DATAFRAME_SOURCE) $(TENSOR_SOURCES) $(HEADERS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) -O2 $(MNIST_SOURCE) $(DATAFRAME_SOURCE) $(TENSOR_SOURCES) -o $(MNIST_BINARY)
