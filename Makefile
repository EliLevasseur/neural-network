CXX := g++
CXXFLAGS := -std=c++17 -g3 -O0 -Wall -Wextra -Wpedantic -Iinclude

.DEFAULT_GOAL := build

SOURCES := src/dataframe.cpp src/network.cpp src/training.cpp
HEADERS := $(wildcard include/*.h)
TEST_SOURCES := tests/test_main.cpp tests/network_test.cpp tests/training_test.cpp

.PHONY: build run test graph

build: build/nnet

run: build/nnet
	clear
	./build/nnet

test: build/nnet_tests
	clear
	./build/nnet_tests

graph: build/nnet
	clear
	./build/nnet --loss-csv | python3 ./visualizations/graphing.py

build/nnet: main.cpp $(SOURCES) $(HEADERS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) main.cpp $(SOURCES) -o build/nnet

build/nnet_tests: $(TEST_SOURCES) tests/test_utils.h $(SOURCES) $(HEADERS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) -Itests $(TEST_SOURCES) $(SOURCES) -o build/nnet_tests
