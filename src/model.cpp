#include <iostream>
#include "model.hpp"

Model::Model(std::vector<Layers::Layer *> &_layers) {
    this->L = (int) _layers.size();
    this->layers = _layers;
}

void Model::compile(Losses::Loss *_loss, Optimizers::Optimizer *_optimizer) {
    this->loss = _loss;
    this->optimizer = _optimizer;
}

void Model::fit(const array &X_train, const array &Y_train, int num_epochs) {
    optimizer->optimize(this, X_train, Y_train, num_epochs);
}

array Model::predict(const array &X_test) {
    auto thread_cache = std::vector<std::unordered_map<std::string, array>>(L);
    return forward(X_test, thread_cache);
}

std::vector<Layers::Layer *> &Model::get_layers() {
    return this->layers;
}

array Model::forward(const array &input, std::vector<std::unordered_map<std::string, array>> &thread_cache) {
    array y_pred = input;

//    std::cout << "\n\n\n\n\n";
    for (int l = 0; l < L; ++l) {
//        af_print(y_pred.cols(0, 50));
        y_pred = layers[l]->forward(y_pred, thread_cache[l]);
    }

    return y_pred;
}

array Model::compute_cost(const array &y_pred, const array &y_true) {

    array losses = loss->loss(y_pred, y_true);
    return af::mean(losses, 0);
}

void Model::backward(const array &y_pred, const array &y_true,
                     std::vector<std::unordered_map<std::string, array>> &thread_cache) {

    array dA = this->loss->loss_back(y_pred, y_true);
    for (int l = L - 1; l >= 0; --l) {

        dA = layers[l]->backward(dA, thread_cache[l]);
    }
}
