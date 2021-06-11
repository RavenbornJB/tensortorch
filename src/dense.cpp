//
// Created by bohdan on 11.06.21.
//

#include <random>
#include <iostream>
#include "layer.hpp"

Layers::Dense::Dense(int from_size, int to_size, Activations::Activation *activation,
                     const std::string &parameter_initialization) {

    this->description.emplace_back("dense");
    this->W = af::constant(0, to_size, from_size);
    this->b = af::constant(0, to_size, 1);
    this->gradients = {"dW", "db"};

    this->activation = activation;
    this->description.push_back(activation->name);

    std::default_random_engine gen{static_cast<long unsigned int>(time(nullptr))};
    double stddev;
    if (parameter_initialization == "normal") {
        stddev = 1;
    } else if (parameter_initialization == "he") {
        stddev = std::sqrt(2. / from_size);
    } else if (parameter_initialization == "xavier") {
        stddev = std::sqrt(6. / (from_size + to_size));
    } else return;

    std::normal_distribution<double> dist(0, stddev);

    for (int i = 0; i < to_size; ++i) {
        for (int j = 0; j < from_size; ++j) {
            this->W(i, j) = dist(gen);
        }
    }

}

array Layers::Dense::forward(const array &input, std::unordered_map<std::string, array> &cache) {

    cache["A_prev"] = input;
    cache["Z"] = af::matmul(W, input) + af::tile(b, 1, input.dims(1));

    return activation->activate(cache["Z"]);
}


array Layers::Dense::backward(const array &dA, std::unordered_map<std::string, array> &cache) {

    array dZ = activation->activate_back(dA, cache["Z"]);

    cache["dW"] = af::matmul(dZ, (cache["A_prev"].T())) / dZ.dims(0);



    cache["db"] = af::sum(W, 1) / W.dims(1);
//    std::cout << "HERE" << std::endl;

    return af::matmul(W.T(), dZ);
}


void Layers::Dense::update_parameters(std::unordered_map<std::string, array> &cache) {
    W -= cache["dW"];
    b -= cache["db"];
}


std::unordered_map<std::string, std::vector<int>> Layers::Dense::layer_shapes() {
    return {
            {"dW", {(int) W.dims(0), (int) W.dims(1)}},
            {"db", {(int) b.dims(0), (int) b.dims(1)}}
    };
}