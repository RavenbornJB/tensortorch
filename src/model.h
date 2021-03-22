//
// Created by raven on 3/22/21.
//

#ifndef NN_PROJECT_MODEL_H
#define NN_PROJECT_MODEL_H

#include <iostream>
#include <vector>


class Model {
private:
    size_t L;
    std::vector<size_t> layer_dims;

public:
    explicit Model(std::vector<size_t> &layer_dims);
    void print_layers();
};


#endif //NN_PROJECT_MODEL_H
