//
// Created by bohdansydor on 13.05.21.
//
#include "optimizers.hpp"
#include "model.hpp"


Optimizers::BGD::BGD(double _learning_rate) {
    this->learning_rate = _learning_rate;
}


void Optimizers::BGD::optimize(Model *model, const MatrixXd &X_train, const MatrixXd &Y_train, int num_epochs) {

    auto thread_cache = std::vector<std::unordered_map<std::string, MatrixXd>>(model->get_layers().size());

    for (int i = 0; i < num_epochs; ++i) {
        MatrixXd Y_pred = model->forward(X_train, thread_cache, true);
        double cost = model->compute_cost(Y_pred, Y_train); // can print or something;
        if (i % (num_epochs / 10) == 0 || i == num_epochs - 1) {
            std::cout << "Cost at iteration " << i << ": " << cost << std::endl;
        }
        model->backward(Y_pred, Y_train, thread_cache);

        update_parameters(model->get_layers(), thread_cache);
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
