//
// Created by raven on 3/22/21.
//

#ifndef NN_PROJECT_LAYER_H
#define NN_PROJECT_LAYER_H

#include <iostream>
#include <string>
#include <random>
#include <stdexcept>

#include "linalg.h"

class Layer {
private:
    Matrix<double> W;
    Matrix<double> b;
    Matrix<double> A_prev;
    Matrix<double> Z;
    size_t m;
    double learning_rate;

    Matrix<double> linear(const Matrix<double> &input);
    static Matrix<double> sigmoid(const Matrix<double> &input);
    static Matrix<double> tanh(const Matrix<double> &input);
    static Matrix<double> relu(const Matrix<double> &input);
    Matrix<double>(*activation)(const Matrix<double> &); // TODO try with std::function

    std::vector<Matrix<double>> linear_backward(const Matrix<double> &dZ);
    static Matrix<double> sigmoid_backward(const Matrix<double> &dA, const Matrix<double> &Z);
    static Matrix<double> tanh_backward(const Matrix<double> &dA, const Matrix<double> &Z);
    static Matrix<double> relu_backward(const Matrix<double> &dA, const Matrix<double> &Z);
    Matrix<double>(*activation_backward)(const Matrix<double> &, const Matrix<double> &);

public:
    explicit Layer(const std::string& activation_type, size_t from_size, size_t to_size, double learning_rate);
    void print_parameters();
    Matrix<double> forward(const Matrix<double> &input);
    std::vector<Matrix<double>> backward(const Matrix<double> &dZ);
    void update_parameters(const Matrix<double> &dW, const Matrix<double> &db);
};


#endif //NN_PROJECT_LAYER_H
