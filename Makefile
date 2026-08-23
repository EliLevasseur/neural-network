CXX := g++
CXXFLAGS := -std=c++17 -g3 -O0 -Wall -Wextra -Wpedantic -Iinclude

SOURCES := src/dataframe.cpp src/network.cpp src/training.cpp
HEADERS := $(wildcard include/*.h)
TEST_SOURCES := tests/test_main.cpp tests/network_test.cpp tests/training_test.cpp

.PHONY: run test graph

run: build/nnet
	./build/nnet

test: build/nnet_tests
	./build/nnet_tests

graph: build/nnet
	./build/nnet | python3 ./visualizations/graphing.py

build/nnet: main.cpp $(SOURCES) $(HEADERS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) main.cpp $(SOURCES) -o build/nnet

build/nnet_tests: $(TEST_SOURCES) tests/test_utils.h $(SOURCES) $(HEADERS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) -Itests $(TEST_SOURCES) $(SOURCES) -o build/nnet_tests
