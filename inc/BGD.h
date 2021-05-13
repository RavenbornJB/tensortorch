//
// Created by raven on 5/11/21.
//

// All formulae for advanced optimization algorithms in this file come from
// TODO insert article link, I know which

#ifndef NEURALNET_LIB_BGD_H
#define NEURALNET_LIB_BGD_H

#include <iostream>

#include "layer.h"


class Model;


namespace Optimizers {
    class Optimizer {
    public:
        virtual void optimize(Model* model, const MatrixXd& X_train, const MatrixXd& Y_train, int num_epochs);
    };

    class BGD: public Optimizer {
    private:
        double learning_rate;
    public:
        explicit BGD(double _learning_rate);

        void optimize(Model* model, const MatrixXd& X_train, const MatrixXd& Y_train, int num_epochs) override;

        void update_parameters(std::vector<Layers::Layer *> &layers, std::vector<std::unordered_map<std::string, MatrixXd>> &cache);
    };
}

#endif //NEURALNET_LIB_BGD_H
