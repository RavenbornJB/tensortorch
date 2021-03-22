//
// Created by raven on 3/22/21.
//

#include "model.h"


Model::Model(std::vector<size_t> &layer_dims) {
    this->L = layer_dims.size();
    this->layer_dims = layer_dims;


}

void Model::print_layers() {
    std::cout << "Layers of the network: \n";
    for (auto l: layer_dims) {
        std::cout << l << "  ";
    }
    std::cout << std::endl;
}
