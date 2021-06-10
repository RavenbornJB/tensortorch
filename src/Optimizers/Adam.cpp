//
// Created by raven on 6/10/21.
//

#include "optimizers.hpp"
#include "model.hpp"

Optimizers::Adam::Adam(int _batch_size, double _learning_rate, double _beta1, double _beta2, double epsilon) {
    this->learning_rate = _learning_rate;
    this->beta1 = _beta1;
    this->beta2 = _beta2;
    //if batch_size == 1 and loss == BinaryCrossentropy => -nan
    this->batch_size = _batch_size;
    this->epsilon = epsilon; // to avoid division by zero
}

void Optimizers::Adam::optimize(Model *model, const MatrixXd &X_train, const MatrixXd &Y_train, int num_epochs) {
    std::vector<Layers::Layer *> layers = model->get_layers();

    auto thread_cache = std::vector<std::unordered_map<std::string, MatrixXd>>(model->get_layers().size());

    auto rms_cache = *(new std::vector<std::unordered_map<std::string, MatrixXd>>(layers.size()));
    auto momentum_cache = *(new std::vector<std::unordered_map<std::string, MatrixXd>>(layers.size()));
    for (int l = 0; l < layers.size(); ++l) {
        std::unordered_map<std::string, std::vector<int>> layer_shapes = layers[l]->layer_shapes();
        for (const auto &grad: layers[l]->gradients) {
            rms_cache[l][grad] = MatrixXd::Zero(layer_shapes[grad][0], layer_shapes[grad][1]);
            momentum_cache[l][grad] = MatrixXd::Zero(layer_shapes[grad][0], layer_shapes[grad][1]);
        }
    }

//    double prev_cost =
    for (int i = 0; i < num_epochs; i++) {
        double cost;
        if (model->get_layers()[0]->description[0] == "rnn") { // TODO temp fix, get RNN batches to work globally
            MatrixXd Y_pred = model->forward(X_train, thread_cache);
            cost = model->compute_cost(Y_pred, Y_train); // can print
            std::cout << "Cost at epoch " << i << ": " << cost << std::endl;
            model->backward(Y_pred, Y_train, thread_cache);
            update_parameters(model->get_layers(), thread_cache, momentum_cache, rms_cache, i);
        } else {
            for (int j = 0; j + batch_size < X_train.cols(); j += batch_size) {
                MatrixXd Y_pred = model->forward(X_train.middleCols(j, batch_size), thread_cache);
                cost = model->compute_cost(Y_pred, Y_train.middleCols(j, batch_size)); // can print
                model->backward(Y_pred, Y_train.middleCols(j, batch_size), thread_cache);
                update_parameters(model->get_layers(), thread_cache, momentum_cache, rms_cache, i);
            }
            std::cout << "Cost at epoch " << i << ": " << cost << std::endl;
        }
    }
}


void Optimizers::Adam::update_parameters(
        std::vector<Layers::Layer *> &layers,
        std::vector<std::unordered_map<std::string, MatrixXd>> &cache,
        std::vector<std::unordered_map<std::string, MatrixXd>> &momentum_cache,
        std::vector<std::unordered_map<std::string, MatrixXd>> &rms_cache,
        int epoch)
{
    // this optimizer accounts uses the exponentially weighted average of squares of gradients
    MatrixXd debiased_momentum;
    MatrixXd debiased_rms;
    for (int l = 0; l < layers.size(); ++l) {
        for (const auto &grad: layers[l]->gradients) {
            momentum_cache[l][grad] = beta1 * momentum_cache[l][grad] + (1 - beta1) * cache[l][grad];
            rms_cache[l][grad] = beta2 * rms_cache[l][grad] + (1 - beta2) * cache[l][grad].array().square().matrix();

            debiased_momentum = momentum_cache[l][grad] / (1 - std::pow(beta1, epoch + 1));
            debiased_rms = rms_cache[l][grad] / (1 - std::pow(beta2, epoch + 1));

            cache[l][grad] = learning_rate * (debiased_momentum.array() / (debiased_rms.array() + epsilon).sqrt());
        }
        layers[l]->update_parameters(cache[l]);
    }
}