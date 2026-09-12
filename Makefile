CXX := g++
CXXFLAGS := -std=c++17 -g3 -O0 -Wall -Wextra -Wpedantic -Iinclude

.DEFAULT_GOAL := build

REFERENCE_SOURCE := examples/reference_mlp.cpp
REFERENCE_SOURCES := src/dataframe.cpp src/network.cpp src/training.cpp
REFERENCE_BINARY := build/reference_mlp

TENSOR_TEST_SOURCES := src/core/tensor.cpp src/core/tensor_dataframe.cpp src/core/tensor_ops.cpp

TEST_SOURCES := tests/test_main.cpp tests/network_test.cpp tests/training_test.cpp tests/tensor_test.cpp tests/dataframe_test.cpp tests/tensor_ops_tests.cpp
TEST_BINARY := build/nnet_tests

HEADERS := $(wildcard include/*.h include/nnet/core/*.h)

.PHONY: build run test graph

build: $(REFERENCE_BINARY)

run: $(REFERENCE_BINARY)
	./$(REFERENCE_BINARY)

test: $(TEST_BINARY)
	./$(TEST_BINARY)

graph: $(REFERENCE_BINARY)
	./$(REFERENCE_BINARY) --loss-csv | python3 ./visualizations/graphing.py

$(REFERENCE_BINARY): $(REFERENCE_SOURCE) $(REFERENCE_SOURCES) $(HEADERS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(REFERENCE_SOURCE) $(REFERENCE_SOURCES) -o $(REFERENCE_BINARY)

$(TEST_BINARY): $(TEST_SOURCES) tests/test_utils.h $(REFERENCE_SOURCES) $(TENSOR_TEST_SOURCES) $(HEADERS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) -Itests $(TEST_SOURCES) $(REFERENCE_SOURCES) $(TENSOR_TEST_SOURCES) -o $(TEST_BINARY)
