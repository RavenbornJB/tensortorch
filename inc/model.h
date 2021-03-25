//
// Created by raven on 3/22/21.
//

#ifndef NEURALNET_LIB_MODEL_H
#define NEURALNET_LIB_MODEL_H

#include <iostream>

#include "linalg.h"
#include "layer.h"

typedef std::vector<std::pair<size_t, std::string>> def_layers_vector;

class Model {
private:
    std::vector<Layer> layers;
    size_t L;
    Matrix<double> forward_propagation(const Matrix<double> &X);
    [[nodiscard]] double compute_cost(const Matrix<double> &AL, const Matrix<double> &Y) const;

public:
    explicit Model(size_t num_input, const def_layers_vector &layer_parameters, double learning_rate);
    void print_layer(size_t n) const;
    void print_layers() const;
    void fit(const Matrix<double> &X, const Matrix<double> &Y);
};


#endif //NN_PROJECT_MODEL_H
