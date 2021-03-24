//
// Created by raven on 3/22/21.
//

#include "model.h"


Model::Model(size_t num_input, const def_layers_vector &layer_parameters, double learning_rate)
: L(layer_parameters.size()), learning_rate(learning_rate) {
    layers.emplace_back(layer_parameters[0].second, num_input, layer_parameters[0].first, learning_rate);
    for (int l = 1; l < L; ++l) {
        std::string activation = layer_parameters[l].second;
        layers.emplace_back(activation, layer_parameters[l-1].first, layer_parameters[l].first, learning_rate);
    }
}

void Model::print_layers() const {
    std::cout << "Layers of the network: \n";
    for (int l = 0; l < L; ++l) {
        Layer test = layers[l];
        test.print_parameters();
    }
}
