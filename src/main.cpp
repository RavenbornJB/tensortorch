#include <iostream>
#include <fstream>

#include "linalg.h"
#include "layer.h"
#include "model.h"


typedef
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
datatype;

datatype get_data(const std::string &filename) {
    std::ifstream datafile(filename);
    if (!datafile.is_open()) throw std::invalid_argument("Incorrect file name");
    size_t num_points;
    datafile >> num_points;

    datatype data;
    data.first.emplace_back();
    data.first.emplace_back();
    data.second.emplace_back();
    double x, y;
    for (size_t i = 0; i < num_points; ++i) {
//        if (i >= 5) {
//            continue;
//        }
        datafile >> x >> y;
        data.first[0].push_back(x);
        data.first[1].push_back(y);
        data.second[0].push_back(0);
    }
    for (size_t i = 0; i < num_points; ++i) {
//        if (i >= 5) {
//            continue;
//        }
        datafile >> x >> y;
        data.first[0].push_back(x);
        data.first[1].push_back(y);
        data.second[0].push_back(1);
    }
    return data;
}


double compare(const mdb &prediction, const mdb &Y) {
    size_t m = Y.get_cols();
    double coincide = 0.;
    for (size_t i = 0; i < m; ++i) {
        coincide += (double)(prediction(0, i) == Y(0, i));
    }
    return coincide / m;
}


int main() {
    return 0;
}