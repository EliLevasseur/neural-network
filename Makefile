CXX := g++
CXXFLAGS := -std=c++17 -g3 -O0 -Wall -Wextra -Wpedantic -Iinclude

.DEFAULT_GOAL := build

REFERENCE_SOURCE := examples/reference_mlp.cpp
REFERENCE_SOURCES := src/dataframe.cpp src/network.cpp src/training.cpp
REFERENCE_BINARY := build/reference_mlp

NETWORK_SOURCES := src/network.cpp src/training.cpp
DATAFRAME_SOURCE := src/dataframe.cpp
TENSOR_SOURCES := src/core/tensor.cpp src/core/tensor_dataframe.cpp src/core/tensor_ops.cpp

REFERENCE_TEST_SOURCES := tests/reference_test_main.cpp tests/network_test.cpp tests/training_test.cpp
REFERENCE_TEST_BINARY := build/reference_tests

TENSOR_TEST_SOURCES := tests/tensor_test_main.cpp tests/tensor_test.cpp tests/dataframe_test.cpp tests/tensor_ops_tests.cpp
TENSOR_TEST_BINARY := build/tensor_tests

HEADERS := $(wildcard include/*.h include/nnet/core/*.h)

.PHONY: build run test test-reference test-tensor graph

build: $(REFERENCE_BINARY)

run: $(REFERENCE_BINARY)
	./$(REFERENCE_BINARY)

test: test-reference test-tensor

test-reference: $(REFERENCE_TEST_BINARY)
	./$(REFERENCE_TEST_BINARY)

test-tensor: $(TENSOR_TEST_BINARY)
	./$(TENSOR_TEST_BINARY)

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
