#include <iostream>
#include <cmath>
#include <vector>
#include "dataframe.h"


struct Node {
    std::vector<double> inputs;
    double output;
};

// Create a data type to representa layer of neurons
struct Layer {
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
};

// KIND OF ERROR BUT I NEED TO MAKE IT OUTPUT THE OUTPUT FOR ONE ROW OF THE CSV NOT THE ENTIRE CSV FILE
double feedForward(std::vector<double>& inputs, Layer& layer, int weightIndex) {
    double output = 0.0;
    for (int i = 0; i < inputs.size(); i++) {
        output += inputs[i] * layer.weights[weightIndex][i];
    }
    return output + layer.biases[weightIndex];
}

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

//nodeMapping

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

int main() { 
    //csv file turned to matrix of vectors of doubles
    std::vector<std::vector<double>> dataFrame = dataframe("data.csv");
    //create a target vector
    std::vector<double> targets;
    targets.reserve(dataFrame.size());
    for (int i = 0; i < dataFrame.size(); i++) {
        targets.push_back(dataFrame[i].back());
        dataFrame[i].pop_back();
    }
    // IDEA IN DEV !! std::vector<int> hiddenLayerSizes = {2, 3, 3, 4};
    Layer test_layer = createLayer(4, 2);
    //Output of each hidden layer to be passed to the next layer as input
    std::vector<double> hiddenLayerOutputs;

  //===============Feedforward Test==========//
  std::cout << std::endl << "Feedforward Test" << std::endl;
  for (auto& row: dataFrame) {
    for (std::size_t index = 0; index < test_layer.weights.size(); ++index) {
        hiddenLayerOutputs.push_back(feedForward(row, test_layer, index));
    }
  }


  //check the output of the feedforward function
  std::cout << "Feedforward Output:" << std::endl;
  for (const auto& val: hiddenLayerOutputs) {
      std::cout << val << " ";
  }
  std::cout << std::endl;

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
  std::cout << "Predictors" << std::endl;
  for (const auto& row: dataFrame) {
      for (const auto& val: row) {
          std::cout << val << " ";
      }
      std::cout << std::endl;
  }
  std::cout << "Targets" << std::endl;
  for (const auto& val: targets) {
        std::cout << val << std::endl;
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

