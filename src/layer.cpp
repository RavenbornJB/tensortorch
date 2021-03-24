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
        throw std::logic_error("Activation type not allowed");
    }
}

/* Debug function for printing W and b.
 */
void Layer::print_parameters() {
    std::cout << "W: " << std::endl;
    W.print();
    std::cout << "\nb: " << std::endl;
    b.print();
}

/* Activation functions.
 */
Matrix<double> Layer::sigmoid(const Matrix<double> &input) {
    Matrix<double> output(input.get_rows(), input.get_cols(), 0);
    for (int i = 0; i < output.get_rows(); ++i) {
        for (int j = 0; j < output.get_cols(); ++j) {
            output(i, j) = 1 / (1 + std::exp(input(i, j)));
        }
    }
    return output;
}
Matrix<double> Layer::tanh(const Matrix<double> &input) {
    Matrix<double> output(input.get_rows(), input.get_cols(), 0);
    for (int i = 0; i < output.get_rows(); ++i) {
        for (int j = 0; j < output.get_cols(); ++j) {
            output(i, j) = std::tanh(input(i, j));
        }
    }
    return output;
}
Matrix<double> Layer::relu(const Matrix<double> &input) {
    Matrix<double> output(input.get_rows(), input.get_cols(), 0);
    for (int i = 0; i < output.get_rows(); ++i) {
        for (int j = 0; j < output.get_cols(); ++j) {
            output(i, j) = input(i, j) * (int)(input(i, j) > 0);
        }
    }
    return output;
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
    Matrix<double> ones (Z.get_rows(), Z.get_cols(), 1);
    return dA * (Z * (ones - Z));
}
Matrix<double> Layer::tanh_backward(const Matrix<double> &dA, const Matrix<double> &Z) {
    Matrix<double> res(dA);
    for (int i = 0; i < Z.get_rows(); ++i) {
        for (int j = 0; j < Z.get_cols(); ++j) res(i, j) *= 1 / std::pow(std::cosh(Z(i, j)), 2);
    }
    return res;
}
Matrix<double> Layer::relu_backward(const Matrix<double> &dA, const Matrix<double> &Z) {
    Matrix<double> res(dA);
    for (int i = 0; i < Z.get_rows(); ++i) {
        for (int j = 0; j < Z.get_cols(); ++j) res(i, j) *= (int)(Z(i, j) > 0);
    }
    return res;
}

/* Linear backpropagation function.
 * Calculates dW, db (for gradient descent), and dA_prev (for previous layer).
 */
std::vector<Matrix<double>> Layer::linear_backward(const Matrix<double> &dZ) {
    Matrix<double> dW = dot(dZ, transpose(A_prev)) / m;
    Matrix<double> db = dZ.sum(1) / m;
    Matrix<double> dA_prev = dot(transpose(W), dZ);
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
    // TODO update in-place when -= is shipped in linalg
    W = W - dW * learning_rate;
    b = b - db * learning_rate;
}

//int main() {
//    Layer test_layer("tanh", 5, 3, 0.01);
//    std::vector<std::vector<double>> inp_forw = {{.1, .1, .1}, {.2, .2, .2}, {.3, .3, .3}, {.4, .4, .4}, {.5, .5, .5}};
//    std::vector<std::vector<double>> inp_back = {{.2, .2, .2}, {.3, .3, .3}, {.4, .4, .4}};
//    Matrix<double> in_f(inp_forw);
//    Matrix<double> in_b(inp_back);
//    auto out_f = test_layer.forward(in_f);
//    auto out_b = test_layer.backward(in_b);
//    std::cout << "dW, db, and dA_prev" << std::endl;
//    for (const auto& matrix: out_b) {
//        matrix.print();
//    }
//    std::cout << "W and b before update" << std::endl;
//    test_layer.print_parameters();
//    test_layer.update_parameters(out_b[0], out_b[1]);
//    std::cout << "W and b after update" << std::endl;
//    test_layer.print_parameters();
//}
