//
// Created by raven on 5/4/21.
//

#ifndef NEURALNET_LIB_MODEL_HPP
#define NEURALNET_LIB_MODEL_HPP

#include <fstream>

#include "layer.hpp"
#include "losses.hpp"
#include "optimizers.hpp"


class Model {
private:
    int L;
    std::vector<Layers::Layer*> layers;
    double regularization_parameter;

    Losses::Loss* loss;
    Optimizers::Optimizer* optimizer;
    bool compiled;

public:
    //TODO constructor options
    explicit Model(std::vector<Layers::Layer*> &layers, double regularization_parameter=-1);

    void compile(Losses::Loss* loss, Optimizers::Optimizer* optimizer);

    MatrixXd forward(const MatrixXd &input, std::vector<std::unordered_map<std::string, MatrixXd>> &thread_cache);
    double compute_cost(const MatrixXd &y_pred, const MatrixXd &y_true);
    void backward(const MatrixXd &y_pred, const MatrixXd &y_true, std::vector<std::unordered_map<std::string, MatrixXd>> &thread_cache);

    void fit(const MatrixXd& X_train, const MatrixXd& Y_train, int num_epochs);
    MatrixXd predict(const MatrixXd& X_test);

    void save(const std::string& filename);

    std::vector<Layers::Layer*>& get_layers();
};

#endif //NEURALNET_LIB_MODEL_HPP
