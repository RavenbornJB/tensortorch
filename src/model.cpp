//
// Created by raven on 5/4/21.
//

#include "model.hpp"
#include "optimizers.hpp"


Model::Model(std::vector<Layers::Layer*> &layers) {
    this->L = (int) layers.size();
    this->layers = layers;

    this->compiled = false;
}


MatrixXd Model::forward(const MatrixXd &input, std::vector<std::unordered_map<std::string, MatrixXd>> &thread_cache) {
    if (!compiled) {
        throw std::logic_error("Model is not compiled yet");
    }

    MatrixXd y_pred(input);

    for (int l = 0; l < L; ++l) {
        y_pred = layers[l]->forward(y_pred, thread_cache[l]);
    }

    return y_pred;
}


double Model::compute_cost(const MatrixXd &y_pred, const MatrixXd &y_true) {
    MatrixXd losses = loss->loss(y_pred, y_true);
    return losses.mean();
}


void Model::backward(const MatrixXd &y_pred, const MatrixXd &y_true, std::vector<std::unordered_map<std::string, MatrixXd>> &thread_cache) {
    MatrixXd dA = this->loss->loss_back(y_pred, y_true);
    for (int l = L - 1; l >= 0; --l) {
        dA = layers[l]->backward(dA, thread_cache[l]);
    }
}


void Model::fit(const MatrixXd& X_train, const MatrixXd& Y_train, int num_epochs) {
    //TODO add shapes check
    optimizer->optimize(this, X_train, Y_train, num_epochs);
}


MatrixXd Model::predict(const MatrixXd& X_test) {
    auto thread_cache = std::vector<std::unordered_map<std::string, MatrixXd>>(L);
    return forward(X_test, thread_cache);
}


std::vector<Layers::Layer *>& Model::get_layers() {
    return this->layers;
}


void Model::compile(Losses::Loss* _loss, Optimizers::Optimizer* _optimizer) {
    loss = _loss;
    optimizer = _optimizer;
    compiled = true;
}