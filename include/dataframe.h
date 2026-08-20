#ifndef DATAFRAME_H
#define DATAFRAME_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <random>
#include <numeric>
#include <cassert>

struct splitContainer {
     std::vector<std::vector<double>> XTest;
     std::vector<std::vector<double>> XTrain;

     std::vector<double> yTest;
     std::vector<double> yTrain;

};

class DataFrame {
    public :
        // CONSTRUCTOR

        DataFrame(const std::string& filename);
        
        // GETTERS

        const std::vector<std::vector<double>>& getPredictors() const;
        const std::vector<double>& getTargets() const;
        const splitContainer& trainTestSplit(double splitSize, std::size_t seed = 42);

        // METHODS

        int dfSize() const;
        void printDataFrame(const std::vector<std::vector<double>>& dfRef) const;
        void printDataFrame(const std::vector<double>& dfRef) const;
        void printDataFrame() const;
        
    private:
        void splitDataFrame();
        splitContainer container;
        std::vector<std::vector<double>> df;
        std::vector<double> targets;
        
};
              
#endif
