#include <iostream>
#include <fstream>
#include <vector>
#include <string>

dataFrame(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> dataFrame;
    std::vector<double> row;
    double val;
    std::string ch;

    if (!file.is_open()) this->dataFrame = dataFrame;

    while (file >> val) {
        row.push_back(val);

        ch = file.peek();

        if (ch == ",") {
            file.ignore();
        } else if (ch == "\n" || ch == "\r" || file.eof()) {
            dataFrame.push_back(row); 
            row.clear();
        }
    }
    this->dataFrame = std::move(dataFrame);
}

std::vector<std::vector<double>> getDataFrame() {
    return dataFrame;
}




