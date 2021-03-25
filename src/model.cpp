//
// Created by raven on 3/22/21.
//

#include "model.h"


/* Creates a list of Layer objects according to num_input and layer_parameters.
 * Gives each of them a learning_rate.
 */
Model::Model(size_t num_input, const def_layers_vector &layer_parameters, double learning_rate)
: L(layer_parameters.size()) {
    layers.emplace_back(layer_parameters[0].second, num_input, layer_parameters[0].first, learning_rate);
    for (int l = 1; l < L; ++l) {
        std::string activation = layer_parameters[l].second;
        layers.emplace_back(activation, layer_parameters[l-1].first, layer_parameters[l].first, learning_rate);
    }
}

void Model::print_layer(size_t n) const {
    if (n <= L) throw std::logic_error("Layer " + std::to_string(n) + " doesn't exist");
    std::cout << "Layer " << n << " of the network: \n";
    layers[n].print_parameters();
}

void Model::print_layers() const {
    std::cout << "Layers of the network: \n";
    for (int l = 0; l < L; ++l) {
        layers[l].print_parameters();
    }
}

Matrix<double> Model::forward_propagation(const Matrix<double> &X) {
    Matrix<double> A(X);
    for (auto &l: layers) {
        A = l.forward(A);
    }
    return A;
}

double Model::compute_cost(const Matrix<double> &AL, const Matrix<double> &Y) const {
    // TODO support for multiple output layer activations, other loss functions
    size_t m = Y.get_cols();
    Matrix<double> first = Y * AL.apply(std::log);
    Matrix<double> second = Y.apply([](double x) {return 1 - x; }) * AL.apply([](double x) {return std::log(1 - x); });
    Matrix<double> test = (first + second).sum(1) / -m;
    test.print();
    return 1;
}

void Model::fit(const Matrix<double> &X, const Matrix<double> &Y) {
    auto AL = forward_propagation(X);
    double res = compute_cost(AL, Y);
}
