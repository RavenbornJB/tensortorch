//
// Created by bohdansydor on 13.05.21.
//
#include "optimizers.h"
#include "model.h"


Optimizers::BGD::BGD(double _learning_rate) {
    this->learning_rate = _learning_rate;
}


void Optimizers::BGD::optimize(Model *model, const MatrixXd &X_train, const MatrixXd &Y_train, int num_epochs) {
//    std::cout << "X_train: " << X_train << std::endl;

    for (int i = 0; i < num_epochs; ++i) {
        MatrixXd Y_pred = model->forward(X_train);
        double cost = model->compute_cost(Y_pred, Y_train); // can print or something;
        if (i % (num_epochs / 10) == 0 || i == num_epochs - 1) {
            std::cout << "Cost at iteration " << i << ": " << cost << std::endl;
        }
        model->backward(Y_pred, Y_train);

        update_parameters(model->get_layers(), model->get_cache());
    }
}


void Optimizers::BGD::update_parameters(std::vector<Layers::Layer *> &layers,
                                        std::vector<std::unordered_map<std::string, MatrixXd>> &cache) {
    for (int l = 0; l < layers.size(); ++l) {
        for (const auto& grad: layers[l]->gradients) {
            cache[l][grad] *= learning_rate;
        }
        layers[l]->update_parameters(cache[l]);
    }
}



