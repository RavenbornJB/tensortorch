//
// Created by raven on 3/22/21.
//

#ifndef NEURALNET_LIB_LAYER_H
#define NEURALNET_LIB_LAYER_H

#include <iostream>
#include <string>
#include <random>
#include <stdexcept>

#include "linalg.h"

typedef Matrix<double> mdb;
typedef std::pair<mdb, mdb> param_pair;

class Layer {
private:
    mdb W;
    mdb b;
    mdb A_prev;
    mdb Z;
    size_t m;
    double learning_rate;

    mdb linear(const mdb &input);
    static mdb sigmoid(const mdb &input);
    static mdb tanh(const mdb &input);
    static mdb relu(const mdb &input);
    mdb(*activation)(const mdb &); // TODO try with std::function

    std::vector<mdb> linear_backward(const mdb &dZ);
    static mdb sigmoid_backward(const mdb &dA, const mdb &Z);
    static mdb tanh_backward(const mdb &dA, const mdb &Z);
    static mdb relu_backward(const mdb &dA, const mdb &Z);
    mdb(*activation_backward)(const mdb &, const mdb &);

public:
    explicit Layer(const std::string& activation_type, size_t from_size, size_t to_size, double learning_rate);
    void print_parameters() const;
    [[nodiscard]] param_pair get_parameters() const;
    mdb forward(const mdb &input);
    std::vector<mdb> backward(const mdb &dA);
    void update_parameters(const mdb &dW, const mdb &db);
};


#endif //NN_PROJECT_LAYER_H
