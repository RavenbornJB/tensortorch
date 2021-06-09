//
// Created by raven on 5/1/21.
//

#include "layer.hpp"

using namespace Layers;

void Dense::constructor(int from_size, int to_size, Activations::Activation* activation_class, const std::string &parameter_initialization) {
    this->description.emplace_back("dense");
    this->W = Eigen::MatrixXd::Zero(to_size, from_size);
    this->b = Eigen::MatrixXd::Zero(to_size, 1);
    this->gradients = {"dW", "db"};

    this->activation = activation_class;
    this->description.push_back(activation_class->name);

    std::default_random_engine gen{static_cast<long unsigned int>(time(nullptr))};
    double stddev;
    if (parameter_initialization == "normal") {
        stddev = 1;
    } else if (parameter_initialization == "he") {
        stddev = std::sqrt(2. / from_size);
    } else if (parameter_initialization == "xavier") {
        stddev = std::sqrt(6. / (from_size + to_size));
    } else {
        throw std::logic_error("Invalid initialization type");
    }

    std::normal_distribution<double> dist(0, stddev); // He initialization

    for (int i = 0; i < to_size; ++i) {
        for (int j = 0; j < from_size; ++j) {
            this->W(i, j) = dist(gen);
        }
    }
}

Dense::Dense(int from_size, int to_size, const std::string &activation_type, const std::string &parameter_initialization) {
    constructor(from_size, to_size, Activations::make_activation(activation_type), parameter_initialization);
}

Dense::Dense(int from_size, int to_size, const std::string &activation_type) {
    constructor(from_size, to_size, Activations::make_activation(activation_type), "normal");
}

Dense::Dense(int from_size, int to_size, Activations::Activation* activation_class, const std::string &parameter_initialization) {
    constructor(from_size, to_size, activation_class, parameter_initialization);
}

Dense::Dense(int from_size, int to_size, Activations::Activation* activation_class) {
    constructor(from_size, to_size, activation_class, "normal");
}

Dense::Dense(int from_size, int to_size) {
    constructor(from_size, to_size, new Activations::Linear, "normal");
}

MatrixXd Dense::linear(const MatrixXd &input) {
//    std::cout << "X_train: " << std::endl;
//    auto a = W * input;
    return (W * input).colwise() + b.col(0); // b is only 1 column, but we use col(0) to transform it to a Vector
}

MatrixXd Dense::forward(const MatrixXd& input, std::unordered_map<std::string, MatrixXd>& cache) {
    cache["A_prev"] = input;
    cache["Z"] = linear(input);
    return activation->activate(cache["Z"]);
}

MatrixXd Dense::linear_backward(const MatrixXd &dZ, std::unordered_map<std::string, MatrixXd>& cache) {
    cache["dW"] = (dZ * cache["A_prev"].transpose()) / dZ.cols();
    cache["db"] = dZ.rowwise().sum() / dZ.cols();
    return W.transpose() * dZ;
}

MatrixXd Dense::backward(const MatrixXd &dA, std::unordered_map<std::string, MatrixXd>& cache) {
    return linear_backward(activation->activate_back(dA, cache["Z"]), cache);
}

void Dense::update_parameters(std::unordered_map<std::string, MatrixXd>& cache) {
    W -= cache["dW"];
    b -= cache["db"];
}

std::unordered_map<std::string, std::vector<int>> Dense::layer_shapes() {
    return {
            {"dW", {(int) W.rows(), (int) W.cols()}},
            {"db", {(int) b.rows(), (int) b.cols()}}
    };
}
