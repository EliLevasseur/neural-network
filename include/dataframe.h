#ifndef DATAFRAME_H
#define DATAFRAME_H

#include <string>
#include <vector>

class DataFrame {
    public :
        // CONSTRUCTOR

        DataFrame(const std::string& filename);
        
        // GETTERS

        const std::vector<std::vector<double>>& getPredictors() const;
        const std::vector<double>& getTargets() const;

        // METHODS

        const int dfSize() const;
        void printDataFrame(const std::vector<std::vector<double>>& dfRef) const;
        void printDataFrame(const std::vector<double>& dfRef) const;
        void printDataFrame() const;
        
    private:
        void splitDataFrame();
        std::vector<std::vector<double>> df;
        std::vector<double> targets;
        
};
              
#endif
