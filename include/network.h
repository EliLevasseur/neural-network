#ifndef NETWORK_H
#define NETWORK_H

#include <vector>

struct Layer {
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
};

class Network {
    public :
        // Constructors
        Network(std::vector<std::size_t> layerSizes);
    
        // GETTERS
        const std::vector<std::vector<double>>& getActivations(const std::vector<double>& inputs);
        std::vector<Layer>& getNetwork();


    private :
        std::vector<Layer> network;
        std::vector<std::vector<double>> activations;
        
        // METHODS
        
        Layer createLayer(int inputs, int nodes);

        double netSummation(const std::vector<double>& inputs, const Layer& layer, int weightIndex);

        double sigmoid(double& x);

        std::vector<double> feedForward(const std::vector<double>& inputs, const Layer& layer); 

        std::vector<double> forwardNetwork(const std::vector<double>& inputs);

        void forwardActivations(const std::vector<double>& inputs);

};
#endif
