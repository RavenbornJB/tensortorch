//
// Created by raven on 6/9/21.
//

#include "layer.hpp"

using namespace Layers;

RNN::RNN(int input_size, int hidden_size, int output_size,
         Activations::Activation *activation_class_a, Activations::Activation *activation_class_y,
         const std::string &parameter_initialization, bool return_sequences) {
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
    this->return_sequences = return_sequences;
    this->parameter_initialization = parameter_initialization;

    this->activation_a = activation_class_a;
    this->activation_y = activation_class_y;
    this->description.push_back(activation_class_a->get_name());
    this->description.push_back(activation_class_y->get_name());

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

void RNN::save(const std::string& filename) {
    std::ofstream file(filename, std::ios::app); // open for append2
    file << description[0] << " " << input_size << " " << hidden_size << " " << output_size << " " <<
    description[1] << " " << description[2] << " " << parameter_initialization << " " << return_sequences << "\n";

    file << Waa << "\n";
    file << Wax << "\n";
    file << Wya << "\n";
    file << ba << "\n";
    file << by << "\n";
}

MatrixXd RNN::cell_forward(const MatrixXd &input, const std::string& timestep,
                           const std::string& batch_num, std::unordered_map<std::string, MatrixXd> &cache) {
    cache["X" + timestep + "_" + batch_num] = input;
    cache["A_prev" + timestep + "_" + batch_num] = hidden_state;
    hidden_state = activation_a->activate(Waa * hidden_state + Wax * input + ba);
    cache["A_next" + timestep + "_" + batch_num] = hidden_state;

    return activation_y->activate(Wya * hidden_state + by);
}

MatrixXd RNN::forward(const MatrixXd &input, std::unordered_map<std::string, MatrixXd> &cache) {
    batch_size = (int) input.rows() / input_size;
    if (batch_size <= 0 || (batch_size > 0 && input.rows() % input_size != 0)) {
        throw std::runtime_error("Invalid RNN input");
    }
    batch_sequence_length = (int) input.cols();

    MatrixXd output(output_size * batch_size, batch_sequence_length);

    MatrixXd relevant_block;
    for (int b = 0; b < batch_size; ++ b) {
        hidden_state = MatrixXd::Zero(hidden_size, 1);

        for (int t = 0; t < batch_sequence_length; ++t) {
            relevant_block = input.block(b * input_size, t, input_size, 1);
            output.block(b * output_size, t, output_size, 1) =
                    cell_forward(relevant_block, std::to_string(t), std::to_string(b), cache);
        }
    }

    if (return_sequences) {
        return output;
    } else {
        return output.col(batch_sequence_length - 1);
    }
}

MatrixXd RNN::linear_backward_y(const MatrixXd &dZ, const std::string& timestep, const std::string& batch_num,
                                std::unordered_map<std::string, MatrixXd>& cache) {
    cache["dWya"] += dZ * cache["A_next" + timestep + "_" + batch_num].transpose();
    cache["dby"] += dZ.rowwise().sum();
    return Wya.transpose() * dZ;
}

MatrixXd RNN::cell_backward_y(const MatrixXd &dA, const std::string& timestep, const std::string& batch_num,
                              std::unordered_map<std::string, MatrixXd>& cache) {
    MatrixXd dZ = activation_y->activate_back(dA, MatrixXd::Zero(0, 0));
    return linear_backward_y(dZ, timestep, batch_num, cache);
}

std::pair<MatrixXd, MatrixXd> RNN::cell_backward_a(const MatrixXd &dA, const std::string &timestep,
                                                   const std::string& batch_num,
                                                   std::unordered_map<std::string, MatrixXd> &cache) {
    MatrixXd dZ = dA.array() * (1 - cache["A_next" + timestep + "_" + batch_num].array().square());

    MatrixXd dX = Wax.transpose() * dZ;
    cache["dWax"] += dZ * cache["X" + timestep + "_" + batch_num].transpose();

    MatrixXd dAprev = Waa.transpose() * dZ;
    cache["dWaa"] += dZ * cache["A_prev" + timestep + "_" + batch_num].transpose();

    cache["dba"] += dZ.rowwise().sum();

    return std::make_pair(dX, dAprev);
}

std::pair<MatrixXd, MatrixXd> RNN::cell_backward(const MatrixXd& dA, const MatrixXd& dA_next,
                                                 const std::string& timestep, const std::string& batch_num,
                                                 std::unordered_map<std::string, MatrixXd>& cache) {
    MatrixXd da = cell_backward_y(dA, timestep, batch_num, cache);
    return cell_backward_a(da + dA_next, timestep, batch_num, cache);
}

MatrixXd RNN::backward(const MatrixXd &dA, std::unordered_map<std::string, MatrixXd> &cache,
                       double regularization_parameter) {
    initialize_gradient_caches(cache);

    MatrixXd output(output_size, batch_sequence_length);
    MatrixXd relevant_block;
    MatrixXd dX(input_size * batch_size, batch_sequence_length);

    for (int b = 0; b < batch_size; ++b) {
        MatrixXd dAprev_t = MatrixXd::Zero(hidden_size, 1);
        MatrixXd dX_t;
        for (int t = batch_sequence_length - 1; t >= 0; --t) {
            if (!return_sequences) {
                if (t == batch_sequence_length - 1) {
                    relevant_block = dA.block(b * output_size, 0, output_size, 1);
                } else {
                    relevant_block = MatrixXd::Zero(output_size, 1);
                }
            } else {
                relevant_block = dA.block(b * output_size, t, output_size, 1);
            }
            std::tie(dX_t, dAprev_t) = cell_backward(relevant_block, dAprev_t, std::to_string(t),
                                                     std::to_string(b), cache);
            dX.block(b * input_size, t, input_size, 1) = dX_t;
        }
    }

    return dX;
}

void RNN::update_parameters(std::unordered_map<std::string, MatrixXd>& cache) {
    Waa -= cache["dWaa"];
    Wax -= cache["dWax"];
    Wya -= cache["dWya"];
    ba -= cache["dba"];
    by -= cache["dby"];
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

void RNN::set_parameters(const std::vector<MatrixXd> &params) {
    Waa = params[0];
    Wax = params[1];
    Wya = params[2];
    ba = params[3];
    by = params[4];
}
