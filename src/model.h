//
// Created by raven on 3/22/21.
//

#ifndef NN_PROJECT_MODEL_H
#define NN_PROJECT_MODEL_H

#include <iostream>
#include <vector>


class Model {
private:
    size_t L;
    std::vector<size_t> layer_dims;
    std::vector<std::vector<std::vector<double>>> W;  // Outside vector over layers, vector of vector inside for matrix.
    std::vector<std::vector<double>> b;

public:
    explicit Model(std::vector<size_t> &layer_dims);
    void print_layers();
};


#endif //NN_PROJECT_MODEL_H
