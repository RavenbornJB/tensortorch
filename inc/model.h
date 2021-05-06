//
// Created by raven on 5/4/21.
//

#ifndef NEURALNET_LIB_MODEL_H
#define NEURALNET_LIB_MODEL_H

#include "layer.h"
#include "losses.h"

class Model {
private:
    int L;
    std::vector<Layer*> layers;
    std::vector<std::unordered_map<std::string, MatrixXd>> cache;

    Losses::Loss* loss;
    std::string optimizer_type;
    std::unordered_map<std::string, double> hparams;

    MatrixXd forward(const MatrixXd &input);
    double compute_cost(const MatrixXd &y_pred, const MatrixXd &y_true);
    void backward(const MatrixXd &y_pred, const MatrixXd &y_true);
    void update_parameters();
public:
    //TODO constructor options
    //TODO move learning rate with other hparams to optimizer
    Model(std::vector<Layer*> &layers, Losses::Loss* loss, const std::string& optimizer_type, double learning_rate);

    void fit(const MatrixXd& X_train, const MatrixXd& Y_train, int num_epochs);
    MatrixXd predict(const MatrixXd& X_test);
};


#endif //NEURALNET_LIB_MODEL_H
