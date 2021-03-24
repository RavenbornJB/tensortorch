//
// Created by raven on 3/22/21.
//

#ifndef NN_PROJECT_MODEL_H
#define NN_PROJECT_MODEL_H

#include <iostream>
#include <layer.h>
#include <linalg.h>

typedef std::vector<std::pair<size_t, std::string>> def_layers_vector;

class Model {
private:
    std::vector<Layer> layers;
    size_t L;
    double learning_rate;

public:
    explicit Model(size_t num_input, const def_layers_vector &layer_parameters, double learning_rate);
    void print_layers() const;
};


#endif //NN_PROJECT_MODEL_H
