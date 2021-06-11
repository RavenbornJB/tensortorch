//
// Created by bohdan on 11.06.21.
//


#include <iostream>
#include "optimizers.hpp"
#include "model.hpp"

void Optimizers::Optimizer::optimize(Model *model, const array &X_train, const array &Y_train, int num_epochs) {};

void Optimizers::Optimizer::update_parameters(std::vector<Layers::Layer *> &layers,
                                              std::vector<std::unordered_map<std::string, array>> &cache,
                                              std::vector<std::unordered_map<std::string, array>> &rms_cache) {};

Optimizers::BGD::BGD(double _learning_rate) {
    this->learning_rate = _learning_rate;
}

void Optimizers::BGD::optimize(Model *model, const array &X_train, const array &Y_train, int num_epochs) {

    auto thread_cache = std::vector<std::unordered_map<std::string, array>>(model->get_layers().size());


    for (int i = 0; i < num_epochs; ++i) {

        array Y_pred = model->forward(X_train, thread_cache);

        array cost = model->compute_cost(Y_pred, Y_train); // can print or something;
        if (i % (num_epochs / 10) == 0 || i == num_epochs - 1) {
            std::cout << "Cost at iteration " << i << std::endl;
        }

        model->backward(Y_pred, Y_train, thread_cache);


        update_parameters(model->get_layers(), thread_cache);
    }
}

void Optimizers::BGD::update_parameters(std::vector<Layers::Layer *> &layers,
                                        std::vector<std::unordered_map<std::string, array>> &cache) {
    for (int l = 0; l < layers.size(); ++l) {
        for (const auto& grad: layers[l]->gradients) {
            cache[l][grad] *= learning_rate;
        }
        layers[l]->update_parameters(cache[l]);
    }
}
