#include "optimizers.hpp"
#include "model.hpp"


Optimizers::SGD::SGD(int _batch_size, double _learning_rate, double _momentum) {
    this->learning_rate = _learning_rate;
    this->momentum = _momentum;
    this->batch_size = _batch_size;
}


void Optimizers::SGD::optimize(Model *model, const MatrixXd &X_train, const MatrixXd &Y_train, int num_epochs) {
    std::vector<Layers::Layer *> layers = model->get_layers();

    auto thread_cache = std::vector<std::unordered_map<std::string, MatrixXd>>(model->get_layers().size());

    auto momentum_cache = *(new std::vector<std::unordered_map<std::string, MatrixXd>>(layers.size()));
    for (int l = 0; l < layers.size(); ++l) {
        std::unordered_map<std::string, std::vector<int>> layer_shapes = layers[l]->layer_shapes();
        for (const auto &grad: layers[l]->gradients) {
            momentum_cache[l][grad] = MatrixXd::Zero(layer_shapes[grad][0], layer_shapes[grad][1]);
        }
    }

    double cost;
    for (int i = 0; i < num_epochs; i++) {
        if (model->get_layers()[0]->description[0] == "rnn") { // TODO temp fix, get RNN batches to work globally
            MatrixXd Y_pred = model->forward(X_train, thread_cache);
            cost = model->compute_cost(Y_pred, Y_train); // can print
            std::cout << "Cost at epoch " << i << ": " << cost << std::endl;
            model->backward(Y_pred, Y_train, thread_cache);
            update_parameters(model->get_layers(), thread_cache, momentum_cache);
        } else {
            for (int j = 0; j + batch_size < X_train.cols(); j += batch_size) {
                MatrixXd Y_pred = model->forward(X_train.middleCols(j, batch_size), thread_cache);
                cost = model->compute_cost(Y_pred, Y_train.middleCols(j, batch_size)); // can print
                model->backward(Y_pred, Y_train.middleCols(j, batch_size), thread_cache);
                update_parameters(model->get_layers(), thread_cache, momentum_cache);
            }
            std::cout << "Cost at epoch " << i << ": " << cost << std::endl;
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
