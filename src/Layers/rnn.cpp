//
// Created by raven on 6/9/21.
//

#include "layer.hpp"

using namespace Layers;

RNN::RNN(int input_size, int hidden_size, int output_size, Activations::Activation *activation_class_a,
         Activations::Activation *activation_class_y, const std::string &parameter_initialization, int sequence_size) {
    this->description.emplace_back("rnn");
    this->Waa = Eigen::MatrixXd::Zero(hidden_size, hidden_size);
    this->Wax = Eigen::MatrixXd::Zero(hidden_size, input_size);
    this->Wya = Eigen::MatrixXd::Zero(output_size, hidden_size);
    this->ba = Eigen::MatrixXd::Zero(hidden_size, 1);
    this->by = Eigen::MatrixXd::Zero(output_size, 1);
    this->gradients = {"dWaa", "dWax", "dWya", "dba", "dby"};

    this->input_size = input_size;
    this->hidden_size = hidden_size;
    this->output_size = output_size;
    this->sequence_size = sequence_size;

    this->activation_a = activation_class_a;
    this->activation_y = activation_class_y;
    this->description.push_back(activation_class_a->name);
    this->description.push_back(activation_class_y->name);

    std::default_random_engine gen{static_cast<long unsigned int>(time(nullptr))};
    double stddev_aa, stddev_ax, stddev_ya;
    if (parameter_initialization == "normal") {
        stddev_aa = 1;
        stddev_ax = 1;
        stddev_ya = 1;
    } else if (parameter_initialization == "he") {
        stddev_aa = std::sqrt(2. / hidden_size);
        stddev_ax = std::sqrt(2. / input_size);
        stddev_ya = std::sqrt(2. / hidden_size);
    } else if (parameter_initialization == "xavier") {
        stddev_aa = std::sqrt(6. / (hidden_size + hidden_size));
        stddev_ax = std::sqrt(6. / (input_size + hidden_size));
        stddev_ya = std::sqrt(6. / (hidden_size + output_size));
    } else {
        throw std::logic_error("Invalid initialization type");
    }

    std::normal_distribution<double> dist_aa(0, stddev_aa);
    std::normal_distribution<double> dist_ax(0, stddev_ax);
    std::normal_distribution<double> dist_ya(0, stddev_ya);

    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            this->Waa(i, j) = dist_aa(gen);
        }
    }
    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            this->Wax(i, j) = dist_ax(gen);
        }
    }
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            this->Wya(i, j) = dist_ya(gen);
        }
    }

    initialized = false;
}

MatrixXd RNN::cell_forward(const MatrixXd &input, MatrixXd &hidden, const std::string& timestep,
                           std::unordered_map<std::string, MatrixXd> &cache) {
    cache["X" + timestep] = input;
    cache["A_prev" + timestep] = hidden;
    hidden = activation_a->activate(Waa * hidden + Wax * input + ba);
    cache["A_next" + timestep] = hidden;

    return activation_y->activate(Wya * hidden + by);
}

MatrixXd RNN::forward(const MatrixXd &input, std::unordered_map<std::string, MatrixXd> &cache) {
    if (input.cols() > sequence_size) {
        throw std::range_error("Input sequence larger than maximum size");
    }
    MatrixXd padded_input(input_size, sequence_size);
    MatrixXd padding = MatrixXd::Zero(input_size, sequence_size - input.cols());
    padded_input << input, padding;

    MatrixXd hidden = MatrixXd::Zero(hidden_size, 1);

    MatrixXd output(output_size, sequence_size);
    for (int t = 0; t < sequence_size; ++t) {
        output.col(t) = cell_forward(padded_input.col(t), hidden, std::to_string(t), cache);
    }

    return output;
}

MatrixXd RNN::linear_backward_y(const MatrixXd &dZ, const std::string& timestep,
                              std::unordered_map<std::string, MatrixXd>& cache) {
    cache["dWya"] += dZ * cache["A_next" + timestep].transpose();
    cache["dby"] += dZ.rowwise().sum();
    return Wya.transpose() * dZ;
}

MatrixXd RNN::cell_backward_y(const MatrixXd &dA, const std::string& timestep,
                                std::unordered_map<std::string, MatrixXd>& cache) {
    MatrixXd dZ = activation_y->activate_back(dA, MatrixXd::Zero(0, 0));
    return linear_backward_y(dZ, timestep, cache);
}

std::pair<MatrixXd, MatrixXd> RNN::cell_backward_a(const MatrixXd &dA, const std::string &timestep,
                              std::unordered_map<std::string, MatrixXd> &cache) {
    MatrixXd dZ = dA.array() * (1 - cache["A_next" + timestep].array().square());

    MatrixXd dX = Wax.transpose() * dZ;
    cache["dWax"] += dZ * cache["X" + timestep].transpose();

    MatrixXd dAprev = Waa.transpose() * dZ;
    cache["dWaa"] += dZ * cache["A_prev" + timestep].transpose();

    cache["dba"] += dZ.rowwise().sum();

    return std::make_pair(dX, dAprev);
}

std::pair<MatrixXd, MatrixXd> RNN::cell_backward(const MatrixXd& dA, const MatrixXd& dA_next,  const std::string& timestep,
                       std::unordered_map<std::string, MatrixXd>& cache) {
    MatrixXd da = cell_backward_y(dA, timestep, cache);
    return cell_backward_a(da + dA_next, timestep, cache);
}

MatrixXd RNN::backward(const MatrixXd &dA, std::unordered_map<std::string, MatrixXd> &cache) {
    initialize_gradient_caches(cache);
    MatrixXd dAprev_t = MatrixXd::Zero(hidden_size, 1);
    MatrixXd dX(input_size, sequence_size);
    MatrixXd dX_t;
    for (int t = sequence_size - 1; t >= 0; --t) {
        std::tie(dX_t, dAprev_t) = cell_backward(dA.col(t), dAprev_t, std::to_string(t), cache);
        dX.col(t) = dX_t;
    }

    return dX;
}

void RNN::update_parameters(std::unordered_map<std::string, MatrixXd>& cache) {
    Waa -= cache["dWaa"] * 0.01;
    Wax -= cache["dWax"] * 0.01;
    Wya -= cache["dWya"] * 0.01;
    ba -= cache["dba"] * 0.01;
    by -= cache["dby"] * 0.01;
}


void RNN::initialize_gradient_caches(std::unordered_map<std::string, MatrixXd> &cache) {
    if (initialized) return;
    for (const auto& grad: layer_shapes()) {
        cache[grad.first] = MatrixXd::Zero(grad.second[0], grad.second[1]);
    }
}

std::unordered_map<std::string, std::vector<int>> RNN::layer_shapes() {
    return {
            {"dWaa", {(int) Waa.rows(), (int) Waa.cols()}},
            {"dWax", {(int) Wax.rows(), (int) Wax.cols()}},
            {"dWya", {(int) Wya.rows(), (int) Wya.cols()}},
            {"dba", {(int) ba.rows(), (int) ba.cols()}},
            {"dby", {(int) by.rows(), (int) by.cols()}}
    };
}

