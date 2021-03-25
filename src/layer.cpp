//
// Created by raven on 3/22/21.
//

#include "layer.h"

/* This initializer fills W with randomly distributed values from N(mean=0, var=2/from_size)
 * It also chooses the activation functions based on the type.
 * */
Layer::Layer(const std::string& activation_type, size_t from_size, size_t to_size, double learning_rate)
: b(*new Matrix<double>(to_size, 1, 0)), learning_rate(learning_rate), m(0) {

    W = *new Matrix<double>(to_size, from_size, 0);
    std::default_random_engine gen{static_cast<long unsigned int>(time(nullptr))};
    std::normal_distribution<double> dist(0., std::sqrt(2. / from_size)); // He initialization
    for (size_t i = 0; i < to_size; ++i) {
        for (size_t j = 0; j < from_size; ++j) {
            W(i, j) = dist(gen);
        }
    }

    // TODO add softmax...
    if (activation_type == "sigmoid") {
        activation = Layer::sigmoid;
        activation_backward = Layer::sigmoid_backward;
    } else if (activation_type == "tanh") {
        activation = Layer::tanh;
        activation_backward = Layer::tanh_backward;
    } else if (activation_type == "relu") {
        activation = Layer::relu;
        activation_backward = Layer::relu_backward;
    } else {
        throw std::logic_error("Activation type " + activation_type + " is not allowed");
    }
}

/* Debug function for printing W and b.
 */
void Layer::print_parameters() const {
    std::cout << "W: " << std::endl;
    W.print();
    std::cout << "b: " << std::endl;
    b.print();
}

param_pair Layer::get_parameters() const {
    return std::make_pair(W, b);
}

/* Activation functions.
 */
Matrix<double> Layer::sigmoid(const Matrix<double> &input) {
    return input.apply([](double x) {return 1 / (1 + std::exp(-x)); });
}
Matrix<double> Layer::tanh(const Matrix<double> &input) {
    return input.apply(std::tanh);
}
Matrix<double> Layer::relu(const Matrix<double> &input) {
    return input.apply([](double x) {return x * (x > 0); });
}

/* Linear forward propagation function: Z = W * X + b
 *
 */
Matrix<double> Layer::linear(const Matrix<double> &input) {
    return dot(W, input) + b;
}

/* Forward-passes the input, first through the linear step,
 * then through the activation function.
 */
Matrix<double> Layer::forward(const Matrix<double> &input) {
    A_prev = input;
    m = input.get_cols();
    Z = linear(input);
    return activation(Z);
}

/* Backward activation functions.
 * */
Matrix<double> Layer::sigmoid_backward(const Matrix<double> &dA, const Matrix<double> &Z) {
    return dA * Z.apply([](double x) {return x * (1 - x); });
}
Matrix<double> Layer::tanh_backward(const Matrix<double> &dA, const Matrix<double> &Z) {
    return dA * Z.apply([](double x) {return 1 / std::pow(std::cosh(x), 2); });
}
Matrix<double> Layer::relu_backward(const Matrix<double> &dA, const Matrix<double> &Z) {
    return dA * Z.apply([](double x) {return (double)(x > 0); });
}

/* Linear backpropagation function.
 * Calculates dW, db (for gradient descent), and dA_prev (for previous layer).
 */
std::vector<Matrix<double>> Layer::linear_backward(const Matrix<double> &dZ) {
    Matrix<double> dW = dot(dZ, A_prev.transpose()) / m;
    Matrix<double> db = dZ.sum(1) / m;
    Matrix<double> dA_prev = dot(W.transpose(), dZ);
    return {dW, db, dA_prev};
}

/* Backward-passes the dA, first through the backward activation,
 * then through the linear step.
 * Calculates dW, db (for gradient descent), and dA_prev (for previous layer).
 */
std::vector<Matrix<double>> Layer::backward(const Matrix<double> &dA) {
    Matrix<double> dZ = activation_backward(dA, Z);
    return linear_backward(dZ);
}

/* Updates W and b based on dW and db according to gradient descent.
 * By default, uses W/b -= learning_rate * dW/b
 */
void Layer::update_parameters(const Matrix<double> &dW, const Matrix<double> &db) {
    W -= dW * learning_rate;
    b -= db * learning_rate;
}
