#include "../include/dataframe.h"

DataFrame::DataFrame(const std::string& filename, const std::size_t targetIndex) : targetIndex(targetIndex) {
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

splitContainer DataFrame::trainTestSplit(double splitSize, std::size_t seed) const {
    if (df.size() < 2)
        throw std::invalid_argument("Training dataset must have more than one row");

    if (splitSize >= 1.0 || splitSize <= 0.0)
        throw std::invalid_argument("Split size must be between one and zero");


    const std::size_t trainCount = static_cast<std::size_t>(splitSize * static_cast<double>(df.size()));

    if (trainCount == 0 || trainCount >= df.size())
        throw std::invalid_argument("Split must have at least one test row and one train row");

    std::vector<std::size_t> indices(df.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::mt19937 generator(seed);

    std::shuffle(indices.begin(), indices.end(), generator);
   
    splitContainer container;

    for (std::size_t row = 0; row < trainCount; row++) {

         container.yTrain.push_back(targets[indices[row]]);
         container.XTrain.push_back(df[indices[row]]);
    }
    
    for (std::size_t row = trainCount; row < df.size(); row++) {
         container.yTest.push_back(targets[indices[row]]);
         container.XTest.push_back(df[indices[row]]);
        }
    return container;
} 

// ======================== METHODS ======================== 

int DataFrame::dfSize() const {
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

    for (std::size_t i = 0; i < df.size(); i++) {
        targets.push_back(df[i][targetIndex]);
        df[i].erase(df[i].begin() + targetIndex);
    }
}


