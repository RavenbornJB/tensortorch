//
// Created by raven on 5/4/21.
//

#include "model.h"
#include "optimizers.h"

Model::Model(std::vector<Layers::Layer*> &layers) {
    this->L = (int) layers.size();
    this->layers = layers;
    this->cache = std::vector<std::unordered_map<std::string, MatrixXd>>(L);

//TODO allow only sigmoid with 1 neuron for binary crossentropy and only softmax with categorical

//    if (this->loss->name == "binary_crossentropy" && this->layers[L - 1]->...) {}
//    if (optimizer_type == "rmsprop" || optimizer_type == "adam" || optimizer_type == "sgd");
}


MatrixXd Model::forward(const MatrixXd &input) {
    MatrixXd y_pred(input);
    for (int l = 0; l < L; ++l) {
        y_pred = layers[l]->forward(y_pred, cache[l]);
    }
    return y_pred;
}


double Model::compute_cost(const MatrixXd &y_pred, const MatrixXd &y_true) {
    MatrixXd losses = loss->loss(y_pred, y_true);
    return losses.mean();
}


void Model::backward(const MatrixXd &y_pred, const MatrixXd &y_true) {
    MatrixXd dA = this->loss->loss_back(y_pred, y_true);
    for (int l = L - 1; l >= 0; --l) {
        dA = layers[l]->backward(dA, cache[l]);
    }
}


void Model::fit(const MatrixXd& X_train, const MatrixXd& Y_train, int num_epochs) {
    optimizer->optimize(this, X_train, Y_train, num_epochs);
}


MatrixXd Model::predict(const MatrixXd& X_test) {
    return forward(X_test);
}


std::vector<Layers::Layer *>& Model::get_layers() {
    return this->layers;
}

std::vector<std::unordered_map<std::string, MatrixXd>>& Model::get_cache() {
    return cache;
}

void Model::compile(Losses::Loss* _loss, Optimizers::Optimizer* _optimizer) {
    loss = _loss;
    optimizer = _optimizer;
}