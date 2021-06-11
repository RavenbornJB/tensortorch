//
// Created by raven on 5/1/21.
//

#include "layer.hpp"

using namespace Layers;

Dense::Dense(int from_size, int to_size, Activations::Activation* activation,
             const std::string &parameter_initialization) {
    this->description.emplace_back("dense");
    this->W = Eigen::MatrixXd::Zero(to_size, from_size);
    this->b = Eigen::MatrixXd::Zero(to_size, 1);
    this->gradients = {"dW", "db"};

    this->input_size = from_size;
    this->output_size = to_size;
    this->parameter_initialization = parameter_initialization;

    this->activation = activation;
    this->description.push_back(activation->get_name());

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

void Dense::save(const std::string& filename) {
    std::ofstream file(filename, std::ios::app); // open for append2
    file << description[0] << " " << input_size << " " << output_size << " " << description[1] << " " <<
    parameter_initialization << "\n";
    file << W << "\n";
    file << b << "\n";
}

MatrixXd Dense::linear(const MatrixXd &input) {
    return (W * input).colwise() + b.col(0); // b is only 1 column, but we use col(0) to transform it to a Vector
}

MatrixXd Dense::forward(const MatrixXd& input, std::unordered_map<std::string, MatrixXd>& cache) {
    cache["A_prev"] = input;
    cache["Z"] = linear(input);
    return activation->activate(cache["Z"]);
}

MatrixXd Dense::linear_backward(const MatrixXd &dZ, std::unordered_map<std::string, MatrixXd>& cache,
                                double regularization_parameter) {
    cache["dW"] = (dZ * cache["A_prev"].transpose()) / dZ.cols();
    cache["db"] = dZ.rowwise().sum() / dZ.cols();

    if (regularization_parameter != -1) {
        cache["dW"] += W * regularization_parameter / dZ.cols();
    }

    return W.transpose() * dZ;
}

MatrixXd Dense::backward(const MatrixXd &dA, std::unordered_map<std::string, MatrixXd>& cache,
                         double regularization_parameter) {
    return linear_backward(activation->activate_back(dA, cache["Z"]), cache, regularization_parameter);
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

void Dense::set_parameters(const std::vector<MatrixXd> &params) {
    W = params[0];
    b = params[1];
}
