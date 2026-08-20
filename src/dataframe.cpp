#include "../include/dataframe.h"

DataFrame::DataFrame(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> df;
    std::vector<double> row;
    double val;
    std::string ch;

    if (!file.is_open()) this->df = df;

    while (file >> val) {
        row.push_back(val);

        ch = file.peek();

        if (ch == ",") {
            file.ignore();
        } else if (ch == "\n" || ch == "\r" || file.eof()) {
            df.push_back(row); 
            row.clear();
        }
    }
    this->df = std::move(df);
    splitDataFrame();
}

// ======================== GETTERS ======================== 

const std::vector<std::vector<double>>& DataFrame::getPredictors() const {
    return df;
}

const std::vector<double>& DataFrame::getTargets() const {
    return targets;
}

// ======================== METHODS ======================== 


const int DataFrame::dfSize() const {
    return df.size();
}

void DataFrame::printDataFrame() const {
    std::cout << "[" << std::endl;
    for (auto& row : df) {
        std::cout << "[ ";
         for (auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "\n]";
}

void DataFrame::printDataFrame(const std::vector<double>& dfRef) const {
    std::cout  << "[ ";
    for (auto& val : dfRef) {
         std::cout << val << " ";
    }
    std::cout << "]\n";
}


void DataFrame::printDataFrame(const std::vector<std::vector<double>>& dfRef) const {
    std::cout << "[" << std::endl;
    for (auto& row : dfRef) {
        std::cout << "[ ";
         for (auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "\n]";
}

void DataFrame::splitDataFrame() {
    targets.reserve(df.size());

    for (int i = 0; i < df.size(); i++) {
        targets.push_back(df[i].back());
        df[i].pop_back();
    }
}

