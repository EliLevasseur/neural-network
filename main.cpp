#include <iostream>
#include <cmath>
#include <vector>
#include "dataframe.h"

// Create a data type to representa layer of neurons
struct Layer {
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
};

Layer createLayer(int inputs, int nodes) {
    Layer layer;

    layer.weights.resize(nodes, std::vector<double>(inputs));
    layer.biases.resize(nodes, 0.0);

    for (auto& nodeWeights: layer.weights) {
        for (auto& weight: nodeWeights) {
            weight = ((double)rand() / RAND_MAX) - 0.5;
        }
        
    }
    return layer;
}

double sigmoid(double x) {
    x = 1.0/(1.0 + std::exp(-x));
    return x;
}

std::vector<std::vector<double>> hiddenLayer(
    const Layer& layer, const std::vector<std::vector<double>>& df) {

    std::vector<std::vector<double>> outputMatrix;

    for (int row = 0; row < df.size(); row++) {
        std::vector<double> outputLayer;
        for (int node = 0; node < layer.weights.size(); node++) {
            double output = 0.0;
            for (int index = 0; index < df[row].size(); index++) {
                output += df[row][index] * layer.weights[node][index];

            }
            output += layer.biases[node];
            output = sigmoid(output);
            outputLayer.push_back(output);
        }

        outputMatrix.push_back(outputLayer);
    }

    return outputMatrix;
}


void Activation(std::vector<double>&arr) {
    for (auto& x: arr) {
        x = sigmoid(x);
    }
}

int main() {  
    std::vector<std::vector<double>> dataFrame = dataframe("data.csv");
    std::vector<Layer> hiddenLayers;
    std::vector<int> hiddenLayerSizes = {2, 3, 3, 4};
    const Layer test_layer = createLayer(3, 5);

    hiddenLayers.resize(hiddenLayerSizes.size());
    
    for (const int& x: hiddenLayerSizes) {
        hiddenLayers.push_back(createLayer(dataFrame[0].size(), x));
    }

  //===============Activation test==========/
  std::vector<double> arr = {1.9, 3.1, 2.4};   
  Activation(arr);
  for (const auto& x: arr) {
    std::cout << x << " ";
  }
  std::cout << "\n";


  //===============Layer creation Test==========//
  std::cout << std::endl << "Layer creation Test" << std::endl;
  for (const auto& row: test_layer.weights) {
    for (const auto& val: row) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
  }

  for (auto& val: test_layer.biases) {
    std::cout << val << std::endl;
  }
  //===============File Read test==========//
  std::cout << "File Read test" << std::endl;
  for (const auto& row: dataFrame) {
      for (const auto& val: row) {
          std::cout << val << " ";
      }
      std::cout << std::endl;
  }

  //===============Hidden Layer Testt==========//
  std::cout << "Hidden Layer Test" << std::endl;
  for (const auto& row: hiddenLayer(test_layer, dataFrame)) {
    for (const auto& val: row) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
  }
  
  return 0;
}

