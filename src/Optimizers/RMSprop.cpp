//
// Created by bohdansydor on 13.05.21.
//
#include "optimizers.hpp"
#include "model.hpp"

Optimizers::RMSprop::RMSprop(int _batch_size,  double _learning_rate, double _beta, double epsilon) {
    this->learning_rate = _learning_rate;
    this->beta = _beta;
    //if batch_size == 1 and loss == BinaryCrossentropy => -nan
    this->batch_size = _batch_size;
    this->epsilon = epsilon; // to avoid division by zero
}

void Optimizers::RMSprop::optimize(Model *model, const MatrixXd &X_train, const MatrixXd &Y_train, int num_epochs) {
    std::vector<Layers::Layer *> layers = model->get_layers();

    auto thread_cache = std::vector<std::unordered_map<std::string, MatrixXd>>(model->get_layers().size());

    auto rms_cache = *(new std::vector<std::unordered_map<std::string, MatrixXd>>(layers.size()));
    for (int l = 0; l < layers.size(); ++l) {
        std::unordered_map<std::string, std::vector<int>> layer_shapes = layers[l]->layer_shapes();
        for (const auto &grad: layers[l]->gradients) {
            rms_cache[l][grad] = MatrixXd::Zero(layer_shapes[grad][0], layer_shapes[grad][1]);
        }
    }

    double cost;
    for (int i = 0; i < num_epochs; i++) {
        if (model->get_layers()[0]->description[0] == "rnn") { // TODO temp fix, get RNN batches to work globally
            MatrixXd Y_pred = model->forward(X_train, thread_cache);
            cost = model->compute_cost(Y_pred, Y_train); // can print
            std::cout << "Cost at epoch " << i << ": " << cost << std::endl;
            model->backward(Y_pred, Y_train, thread_cache);
            update_parameters(model->get_layers(), thread_cache, rms_cache);
        } else {
            for (int j = 0; j + batch_size < X_train.cols(); j += batch_size) {
                MatrixXd Y_pred = model->forward(X_train.middleCols(j, batch_size), thread_cache);
                cost = model->compute_cost(Y_pred, Y_train.middleCols(j, batch_size)); // can print
                model->backward(Y_pred, Y_train.middleCols(j, batch_size), thread_cache);
                update_parameters(model->get_layers(), thread_cache, rms_cache);
            }
            std::cout << "Cost at epoch " << i << ": " << cost << std::endl;
        }
    }
}


void Optimizers::RMSprop::update_parameters(
        std::vector<Layers::Layer *> &layers,
        std::vector<std::unordered_map<std::string, MatrixXd>> &cache,
        std::vector<std::unordered_map<std::string, MatrixXd>> &rms_cache)
{
    // this optimizer accounts uses the exponentially weighted average of squares of gradients
    for (int l = 0; l < layers.size(); ++l) {
        for (const auto &grad: layers[l]->gradients) {
            rms_cache[l][grad] = beta * rms_cache[l][grad] + (1 - beta) * cache[l][grad].array().square().matrix();
            cache[l][grad] = learning_rate * (cache[l][grad].array() / (rms_cache[l][grad].array() + epsilon).sqrt());
        }
        layers[l]->update_parameters(cache[l]);
    }
}
