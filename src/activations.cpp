//
// Created by raven on 6/9/21.
//

#include "activations.hpp"

using namespace Activations;

Activation* make_activation(const std::string& activation_type) {
    if (activation_type == "linear") {
        return new Linear;
    } else if (activation_type == "sigmoid") {
        return new Sigmoid;
    } else if (activation_type == "softmax") {
        return new Softmax;
    } else if (activation_type == "tanh") {
        return new Tanh;
    } else if (activation_type == "relu") {
        return new Relu;
    } else {
        throw std::logic_error("Activation type " + activation_type + " is not allowed");
    }
}