#ifndef DATAFRAME_H
#define DATAFRAME_H

#include <string>
#include <vector>

class DataFrame {
    public :
        DataFrame(const std::string& filename);

        std::vector<std::vector<double>>& getDataFrame();
        const std::vector<double> getTargets();
        const int dfSize() const;
        void printDataFrame() const;
        

    private:
        std::vector<std::vector<double>> df;
};
              
#endif
