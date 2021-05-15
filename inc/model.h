//
// Created by raven on 5/4/21.
//

#ifndef NEURALNET_LIB_MODEL_H
#define NEURALNET_LIB_MODEL_H

#include "layers.h"
#include "losses.h"
#include "optimizers.h"


class Model {
private:
    int L;
    std::vector<Layers::Layer*> layers;

    Losses::Loss* loss;
    Optimizers::Optimizer* optimizer;

public:
    //TODO constructor options
    Model(std::vector<Layers::Layer*> &layers);

    void compile(Losses::Loss* loss, Optimizers::Optimizer* optimizer);

    void fit(const MatrixXd& X_train, const MatrixXd& Y_train, int num_epochs);

    MatrixXd predict(const MatrixXd& X_test);

    std::vector<Layers::Layer*>& get_layers();

    MatrixXd forward(const MatrixXd &input, std::vector<std::unordered_map<std::string, MatrixXd>> &thread_cache);

    double compute_cost(const MatrixXd &y_pred, const MatrixXd &y_true);

    void backward(const MatrixXd &y_pred, const MatrixXd &y_true, std::vector<std::unordered_map<std::string, MatrixXd>> &thread_cache);
};

#endif //NEURALNET_LIB_MODEL_H
