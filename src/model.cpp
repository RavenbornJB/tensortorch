//
// Created by raven on 5/4/21.
//

#include <iostream>

#include "model.h"


Model::Model(std::vector<Layer*> &layers, Losses::Loss* loss, const std::string& optimizer_type, double learning_rate) {
    this->L = (int) layers.size();
    this->layers = layers;
    this->cache = std::vector<std::unordered_map<std::string, MatrixXd>>(L);

    this->loss = loss; //TODO consider move
    this->optimizer_type = std::string(optimizer_type);
    std::for_each(this->optimizer_type.begin(), this->optimizer_type.end(), [](char &c) {
        c = (char) std::tolower(c);
    });
    this->hparams["a"] = learning_rate;

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
    return -losses.mean();
}

void Model::backward(const MatrixXd &y_pred, const MatrixXd &y_true) {
    MatrixXd dA = this->loss->loss_back(y_pred, y_true);
    for (int l = L - 1; l >= 0; --l) {
        dA = layers[l]->backward(dA, cache[l]);
    }
}

void Model::update_parameters() {
    //TODO create optimizer
//    this.optimizer.update_parameters(layers, cache);
    for (int l = 0; l < L; ++l) {
        layers[l]->update_parameters(hparams, cache[l]);
    }
}

void Model::fit(const MatrixXd& X_train, const MatrixXd& Y_train, int num_epochs) {
    for (int i = 0; i < num_epochs; ++i) {
        //TODO SMALL implement mini-batching
        //TODO LARGE refactor with optimizer
        MatrixXd Y_pred = forward(X_train);
//        double cost = compute_cost(Y_pred, Y_train); // can print or something;
        backward(Y_pred, Y_train);
        update_parameters();
    }
}

MatrixXd Model::predict(const MatrixXd& X_test) {
    return forward(X_test);
}