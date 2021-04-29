//
// Created by raven on 3/22/21.
//

#include "layer.h"


/* This initializer fills W with randomly distributed values from N(mean=0, var=2/from_size)
 * It also chooses the activation functions based on the type.
 * */
Layer::Layer(const std::string& activation_type, size_t from_size, size_t to_size, double learning_rate)
: b(*new mdb(to_size, 1, 0)), learning_rate(learning_rate), m(0) {

    W = *new mdb(to_size, from_size, 0);
    std::default_random_engine gen{static_cast<long unsigned int>(time(nullptr))};
//    std::default_random_engine gen{to_size};
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

std::pair<mdb, mdb> Layer::get_parameters() const {
    return std::make_pair(W, b);
}

/* Activation functions.
 */
mdb Layer::sigmoid(const mdb &input) {
    return input.apply([](double x) {return 1 / (1 + std::exp(-x)); });
}
mdb Layer::tanh(const mdb &input) {
    return input.apply(std::tanh);
}
mdb Layer::relu(const mdb &input) {
    return input.apply([](double x) {return std::max(x, 0.); });
}

/* Linear forward propagation function: Z = W * X + b
 *
 */
mdb Layer::linear(const mdb &input) {
    return dot(W, input) + b;
}

/* Forward-passes the input, first through the linear step,
 * then through the activation function.
 */
mdb Layer::forward(const mdb &input) {
    A_prev = input;
    m = input.get_cols();
    Z = linear(input);
    return activation(Z);
}

/* Backward activation functions.
 * */
mdb Layer::sigmoid_backward(const mdb &dA, const mdb &Z) {
    return dA * sigmoid(Z) * sigmoid(Z).apply(one_minus<double>);
}
mdb Layer::tanh_backward(const mdb &dA, const mdb &Z) {
    return dA * Z.apply([](double x) {return 1 / std::pow(std::cosh(x), 2); });
}
mdb Layer::relu_backward(const mdb &dA, const mdb &Z) {
    return dA * Z.apply([](double x) {return (double)(x > 0); });
}

/* Linear backpropagation function.
 * Calculates dW, db (for gradient descent), and dA_prev (for previous layer).
 */
std::vector<mdb> Layer::linear_backward(const mdb &dZ) {
    mdb dW = dot(dZ, A_prev.transpose()) / m;
    mdb db = dZ.sum(1) / m;
    mdb dA_prev = dot(W.transpose(), dZ);
    return {dW, db, dA_prev};
}

/* Backward-passes the dA, first through the backward activation,
 * then through the linear step.
 * Calculates dW, db (for gradient descent), and dA_prev (for previous layer).
 */
std::vector<mdb> Layer::backward(const mdb &dA) {
    mdb dZ = activation_backward(dA, Z);
    return linear_backward(dZ);
}

/* Updates W and b based on dW and db according to gradient descent.
 * By default, uses W/b -= learning_rate * dW/b
 */
void Layer::update_parameters(const mdb &dW, const mdb &db) {
    W -= dW * learning_rate;
    b -= db * learning_rate;
}
