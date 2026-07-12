#include <iostream>
#include <fstream>
#include <vector>
#include <string>

std::vector<std::vector<double>> dataframe(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> dataFrame;
    std::vector<double> row;
    double val;
    std::string ch;

    if (!file.is_open()) return dataFrame;

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
    return dataFrame;
}