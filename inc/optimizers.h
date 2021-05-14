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

    class SGD: public Optimizer {
    private:
        double learning_rate;
        double momentum;
        int batch_size;

    public:
        SGD(int _batch_size, double learning_rate, double momentum);

        void optimize(Model *model, const MatrixXd &X_train, const MatrixXd &Y_train, int num_epochs);

        void update_parameters(
                std::vector<Layers::Layer *> &layers,
                std::vector<std::unordered_map<std::string, MatrixXd>> &cache,
                std::vector<std::unordered_map<std::string, MatrixXd>> &momentum_cache);
    };

    class RMSprop: public Optimizer { // TODO i'm not even sure this works
    private:
        double learning_rate;
        double _batch_size;
        double beta;
        double epsilon;

    public:
        RMSprop(int _batch_size,  double _learning_rate, double _beta,  double epsilon=std::pow(10, -7));

        void optimize(Model *model, const MatrixXd &X_train, const MatrixXd &Y_train, int num_epochs);

        void update_parameters(
                std::vector<Layers::Layer *> &layers,
                std::vector<std::unordered_map<std::string, MatrixXd>> &cache,
                std::vector<std::unordered_map<std::string, MatrixXd>> &rms_cache);
    };

}

#endif //NEURALNET_LIB_BGD_H
