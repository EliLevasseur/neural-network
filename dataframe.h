#ifndef DATAFRAME_H
#define DATAFRAME_H

#include <string>
#include <vector>

class DataFrame {
    public :
        DataFrame();
        DataFrame~();
        DataFrame(const std::string& filename));

        std::vector<std::vector<double>> getDataFrame();

    private:
        std::vector<std::vector<double>> dataFrame;
};
              
#endif
