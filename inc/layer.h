//
// Created by raven on 3/22/21.
//

#ifndef NN_PROJECT_LAYER_H
#define NN_PROJECT_LAYER_H

#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <stdexcept>


class Layer {
private:
    std::vector<std::vector<double>> W;
    std::vector<double> b;

    std::vector<std::vector<double>> linear(const std::vector<std::vector<double>> &input);
    static std::vector<std::vector<double>> sigmoid(const std::vector<std::vector<double>> &input);
    static std::vector<std::vector<double>> tanh(const std::vector<std::vector<double>> &input);
    static std::vector<std::vector<double>> relu(const std::vector<std::vector<double>> &input);
    std::vector<std::vector<double>>(*activation)(const std::vector<std::vector<double>> &); // TODO try with std::function

public:
    explicit Layer(const std::string& activation_type, size_t from_size, size_t to_size);
    void print_parameters();
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>> &input);
    std::vector<std::vector<double>> backward(const std::vector<std::vector<double>> &dA); // TODO зробити
};


#endif //NN_PROJECT_LAYER_H
