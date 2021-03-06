//
// Created by raven on 5/4/21.
//

#ifndef NEURALNET_LIB_LOSSES_HPP
#define NEURALNET_LIB_LOSSES_HPP

#include <iostream>

#include "Dense"

using Eigen::MatrixXd;

namespace Losses { //TODO change inline

    class Loss {
    public:
        virtual inline std::string get_name() { return "abstract"; }
        virtual inline MatrixXd loss(const MatrixXd &y_pred, const MatrixXd &y_true) { return y_pred; }
        virtual inline MatrixXd loss_back(const MatrixXd &y_pred, const MatrixXd &y_true) { return y_pred; }
    };


    class BinaryCrossentropy: public Loss {
    public:
        inline std::string get_name() override { return "binary_crossentropy"; }
        inline MatrixXd loss(const MatrixXd &y_pred, const MatrixXd &y_true) override {
            MatrixXd logprobs = y_true.array() * y_pred.array().log() + (1 - y_true.array()) * (1 - y_pred.array()).log();
//            std::cout << y_pred << std::endl;

            return -logprobs; // generalization for multiple output neurons
        }
        inline MatrixXd loss_back(const MatrixXd &y_pred, const MatrixXd &y_true) override {
            MatrixXd logprobs = y_true.array() / y_pred.array() - (1 - y_true.array()) / (1 - y_pred.array());
            return -logprobs; // generalization for multiple output neurons
        }
    };

    class CategoricalCrossentropy: public Loss {
    public:
        inline std::string get_name() override { return "categorical_crossentropy"; }
        inline MatrixXd loss(const MatrixXd &y_pred, const MatrixXd &y_true) override {
            MatrixXd logprobs = y_true.array() * y_pred.array().log();
            return -logprobs.colwise().sum();
        }
        inline MatrixXd loss_back(const MatrixXd& y_pred, const MatrixXd& y_true) override {
            return y_pred - y_true;
        }
    };

    class MSE: public Loss {
    public:
        inline std::string get_name() override { return "mean_squared_error"; }
        inline MatrixXd loss(const MatrixXd &y_pred, const MatrixXd &y_true) override {
            MatrixXd errors = (y_true.array()  - y_pred.array()).square();
            return errors.colwise().mean();
        }
        inline MatrixXd loss_back(const MatrixXd& y_pred, const MatrixXd& y_true) override {
            MatrixXd errors = -2 * (y_true.array()  - y_pred.array());
            return errors;
        }
    };

}

#endif //NEURALNET_LIB_LOSSES_HPP
