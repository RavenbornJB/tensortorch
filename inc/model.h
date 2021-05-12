//
// Created by raven on 5/4/21.
//

#ifndef NEURALNET_LIB_MODEL_H
#define NEURALNET_LIB_MODEL_H

#include "layer.h"
#include "losses.h"
#include "optimizers.h"

class Model {
private:
    int L;
    std::vector<Layer*> layers;
    std::vector<std::unordered_map<std::string, MatrixXd>> cache;

    Losses::Loss* loss;
    Optimizers::Optimizer* optimizer;

    MatrixXd forward(const MatrixXd &input);
    double compute_cost(const MatrixXd &y_pred, const MatrixXd &y_true);
    void backward(const MatrixXd &y_pred, const MatrixXd &y_true);
public:
    //TODO constructor options
    Model(std::vector<Layer*> &layers, Losses::Loss* loss, Optimizers::Optimizer* optimizer);

    void fit(const MatrixXd& X_train, const MatrixXd& Y_train, int num_epochs);
    MatrixXd predict(const MatrixXd& X_test);
};


#endif //NEURALNET_LIB_MODEL_H
