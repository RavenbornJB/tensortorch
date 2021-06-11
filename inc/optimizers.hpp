// All formulae for advanced optimization algorithms in this file come from
// TODO insert article link, I know which

#ifndef NEURALNET_LIB_BGD_H
#define NEURALNET_LIB_BGD_H

#include <iostream>

#include "layer.hpp"
#include "thread_pool.hpp"

class Model;

namespace Optimizers {

    class Optimizer {
    public:
        virtual inline std::string get_name() { return "abstract"; }
        virtual inline std::vector<double> get_params() {return {}; }
        virtual void optimize(Model* model, const MatrixXd& X_train, const MatrixXd& Y_train, int num_epochs) {};
    };

    class BGD: public Optimizer {
    private:
        double learning_rate;
    public:
        explicit BGD(double _learning_rate);
        inline std::string get_name() override { return "bgd"; }
        inline std::vector<double> get_params() override {return {learning_rate}; }
        void optimize(Model* model, const MatrixXd& X_train, const MatrixXd& Y_train, int num_epochs) override;
        void update_parameters(std::vector<Layers::Layer *> &layers,
                               std::vector<std::unordered_map<std::string, MatrixXd>> &cache);
    };


    class SGD: public Optimizer {
    private:
        double learning_rate;
        double momentum;
        int batch_size;
    public:
        SGD(int _batch_size, double learning_rate, double momentum);
        inline std::string get_name() override { return "sgd"; }
        inline std::vector<double> get_params() override {return {(double) batch_size, learning_rate, momentum}; }
        void optimize(Model *model, const MatrixXd &X_train, const MatrixXd &Y_train, int num_epochs) override;
        void update_parameters(
                std::vector<Layers::Layer *> &layers,
                std::vector<std::unordered_map<std::string, MatrixXd>> &cache,
                std::vector<std::unordered_map<std::string, MatrixXd>> &momentum_cache);
    };

    class RMSprop: public Optimizer {
    private:
        double learning_rate;
        int batch_size;
        double beta;
        double epsilon;
    public:
        RMSprop(int _batch_size,  double _learning_rate, double _beta,  double epsilon=std::pow(10, -7));
        inline std::string get_name() override { return "rmsprop"; }
        inline std::vector<double> get_params() override {return {(double) batch_size, learning_rate, beta}; }
        void optimize(Model *model, const MatrixXd &X_train, const MatrixXd &Y_train, int num_epochs) override;
        void update_parameters(std::vector<Layers::Layer *> &layers,
                std::vector<std::unordered_map<std::string, MatrixXd>> &cache,
                std::vector<std::unordered_map<std::string, MatrixXd>> &rms_cache);
    };

    class Adam: public Optimizer {
    private:
        double learning_rate;
        int batch_size;
        double beta1;
        double beta2;
        double epsilon;
    public:
        Adam(int _batch_size,  double _learning_rate, double _beta1, double _beta2, double epsilon=std::pow(10, -7));
        inline std::string get_name() override { return "adam"; }
        inline std::vector<double> get_params() override {return {(double) batch_size, learning_rate, beta1, beta2}; }
        void optimize(Model *model, const MatrixXd &X_train, const MatrixXd &Y_train, int num_epochs) override;
        void update_parameters(std::vector<Layers::Layer *> &layers,
                               std::vector<std::unordered_map<std::string, MatrixXd>> &cache,
                               std::vector<std::unordered_map<std::string, MatrixXd>> &momentum_cache,
                               std::vector<std::unordered_map<std::string, MatrixXd>> &rms_cache,
                               int epoch);
    };

    class Parallel: public Optimizer {
    private:
        double learning_rate;
        double momentum;
        int batch_size;
    public:
        Parallel(int _batch_size, double learning_rate);
        inline std::string get_name() override { return "parallel"; }
        inline std::vector<double> get_params() override {return {(double) batch_size, learning_rate}; }
        void optimize(Model *model, const MatrixXd &X_train, const MatrixXd &Y_train, int num_epochs) override;
        void update_parameters(std::vector<Layers::Layer *> &layers,
                               std::vector<std::unordered_map<std::string, MatrixXd>> &cache);
    };

}

#endif //NEURALNET_LIB_BGD_H
