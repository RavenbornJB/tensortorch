//
// Created by bohdansydor on 13.05.21.
//

#include "optimizers.h"
#include "model.h"


Optimizers::SGD::SGD(int _batch_size, double _learning_rate, double _momentum) {
    this->learning_rate = _learning_rate;
    this->momentum = _momentum;
    this->batch_size = _batch_size;
}


void Optimizers::SGD::optimize(Model *model, const MatrixXd &X_train, const MatrixXd &Y_train, int num_epochs) {
    std::vector<Layers::Layer *> layers = model->get_layers();

    auto momentum_cache = *(new std::vector<std::unordered_map<std::string, MatrixXd>>(layers.size()));
    for (int l = 0; l < layers.size(); ++l) {
        std::unordered_map<std::string, std::vector<int>> layer_shapes = layers[l]->layer_shapes();
        for (const auto &grad: layers[l]->gradients) {
            momentum_cache[l][grad] = MatrixXd::Zero(layer_shapes[grad][0], layer_shapes[grad][1]);
        }
    }

    for (int i = 0; i < num_epochs; i++) {
        for (int j = 0; j + batch_size < X_train.cols(); j+=batch_size) {
            MatrixXd Y_pred = model->forward(X_train.middleCols(j, batch_size));
            double cost = model->compute_cost(Y_pred, Y_train.middleCols(j, batch_size)); // can print or something;
            if ((i % (num_epochs / 10) == 0 || i == num_epochs - 1) && j==0) {
                std::cout << "Cost at iteration " << i << ": " << cost << std::endl;
            }
            model->backward(Y_pred, Y_train.middleCols(j, batch_size));
            update_parameters(model->get_layers(), model->get_cache(), momentum_cache);
        }

    }
}


void Optimizers::SGD::update_parameters(std::vector<Layers::Layer *> &layers,
                                        std::vector<std::unordered_map<std::string, MatrixXd>> &cache,
                                        std::vector<std::unordered_map<std::string, MatrixXd>> &momentum_cache)
{
    for (int l = 0; l < layers.size(); ++l) {
        for (const auto &grad: layers[l]->gradients) {
            momentum_cache[l][grad] = momentum * momentum_cache[l][grad] + (1 - momentum) * cache[l][grad];
            cache[l][grad] = momentum_cache[l][grad] * learning_rate;
        }
        layers[l]->update_parameters(cache[l]);
    }
}



