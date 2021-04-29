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
    if (n >= L) throw std::logic_error("Layer " + std::to_string(n) + " doesn't exist");
    std::cout << "Layer " << n << " of the network: \n";
    layers[n].print_parameters();
}

void Model::print_layers() const {
    std::cout << "Layers of the network: \n";
    for (int l = 0; l < L; ++l) {
        layers[l].print_parameters();
    }
}

mdb Model::forward_propagation(const mdb &X) {
    mdb A_layer(X);
    for (auto &l: layers) {
        A_layer = l.forward(A_layer);
    }
    return A_layer;
}

double Model::compute_cost(const mdb &AL, const mdb &Y) {
    // TODO support for multiple output layer activations, other loss functions
    size_t m = Y.get_cols();
    mdb logprobs = Y * AL.apply(std::log);
    logprobs += Y.apply(one_minus<double>) * AL.apply([](double x) {return std::log(1 - x); });
    mdb res = logprobs.sum(1) * -1 / m;
    return res.squeeze();
}

void Model::backward_propagation_with_update(const mdb &AL, const mdb &Y) {
    mdb dA_layer = Y / AL;
    dA_layer -= Y.apply(one_minus<double>) / AL.apply(one_minus<double>);
    dA_layer *= -1;
    for (size_t l = 1; l <= L; ++l) {
        auto res = layers[L - l].backward(dA_layer);
        layers[L - l].update_parameters(res[0], res[1]);
        dA_layer = res[2];
    }
}

void Model::fit(const mdb &X, const mdb &Y, size_t num_iters, bool verbose) {
    for (size_t i = 0; i < num_iters; ++i) {
        mdb AL = forward_propagation(X);
        double cost = compute_cost(AL, Y);
        if (verbose && i % (num_iters / 10) == 0) {
            std::cout << "Cost after iteration " << i << ": " << cost << std::endl;
        }
        backward_propagation_with_update(AL, Y);
    }
    if (verbose) {
        mdb AL = forward_propagation(X);
        double cost = compute_cost(AL, Y);
        std::cout << "Cost after iteration " << num_iters << ": " << cost << std::endl;
    }
}

mdb Model::predict(const mdb &X) {
    // TODO make security checks on sizes here and in other places
    mdb prediction = forward_propagation(X);
    prediction.apply_inplace([](double x) {return std::floor(x + 0.5); });
    return prediction;
}
