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
    mdb forward_propagation(const mdb &X);
    [[nodiscard]] static double compute_cost(const mdb &AL, const mdb &Y);
    void backward_propagation_with_update(const mdb &AL, const mdb &Y);

public:
    explicit Model(size_t num_input, const def_layers_vector &layer_parameters, double learning_rate);
    void print_layer(size_t n) const;
    void print_layers() const;
    void fit(const mdb &X, const mdb &Y, size_t num_iters, bool verbose);
    mdb predict(const mdb &X);
};


#endif //NN_PROJECT_MODEL_H
