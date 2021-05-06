//
// Created by raven on 5/1/21.
//

#include "dense.h"


void Dense::constructor(int from_size, int to_size, Activations::Activation* activation_class, const std::string &parameter_initialization) {
    this->W = Eigen::MatrixXd::Zero(to_size, from_size);
    this->b = Eigen::MatrixXd::Zero(to_size, 1);

    this->activation = activation_class;
    this->name = "dense";

    std::default_random_engine gen{static_cast<long unsigned int>(time(nullptr))};
    double stddev;
    if (parameter_initialization == "normal") {
        stddev = 1;
    } else if (parameter_initialization == "he") {
        stddev = std::sqrt(2. / from_size);
    } else if (parameter_initialization == "xavier") {
        stddev = std::sqrt(6. / (from_size + to_size));
    } else return;

    std::normal_distribution<double> dist(0, stddev); // He initialization

    for (int i = 0; i < to_size; ++i) {
        for (int j = 0; j < from_size; ++j) {
            this->W(i, j) = dist(gen);
        }
    }
}

Activations::Activation* Dense::make_activation(const std::string& activation_type) {
    if (activation_type == "linear") {
        return new Activations::Linear;
    } else if (activation_type == "sigmoid") {
        return new Activations::Sigmoid;
    } else if (activation_type == "softmax") {
        return new Activations::Softmax;
    } else if (activation_type == "tanh") {
        return new Activations::Tanh;
    } else if (activation_type == "relu") {
        return new Activations::Relu;
    } else {
        throw std::logic_error("Activation type " + activation_type + " is not allowed");
    }
}

Dense::Dense(int from_size, int to_size, const std::string &activation_type, const std::string &parameter_initialization) {
    constructor(from_size, to_size, make_activation(activation_type), parameter_initialization);
}

Dense::Dense(int from_size, int to_size, const std::string &activation_type) {
    constructor(from_size, to_size, make_activation(activation_type), "normal");
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
    return W * input + b;
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

void Dense::update_parameters(std::unordered_map<std::string, double>& hparams, std::unordered_map<std::string, MatrixXd>& cache) {
    W -= cache["dW"] * hparams["a"];
    b -= cache["db"] * hparams["a"];
}

std::pair<MatrixXd, MatrixXd> Dense::get_parameters() {
    return std::make_pair(W, b);
}

void Dense::set_parameters(const MatrixXd& W_new, const MatrixXd& b_new) {
    if (W_new.rows() == W.rows() && W_new.cols() == W.cols() &&
        b_new.rows() == b.rows() && b_new.cols() == b.cols()) {
        W = MatrixXd(W_new);
        b = MatrixXd(b_new);
    }
}

std::pair<std::vector<int>, std::vector<int>> Dense::get_shape() {
    std::vector<int> input_shape = {(int) W.cols()};
    std::vector<int> output_shape = {(int) W.rows()};
    return std::make_pair(input_shape, output_shape);
}
