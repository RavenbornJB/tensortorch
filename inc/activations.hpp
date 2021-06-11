//
// Created by raven on 5/1/21.
//

#ifndef NEURALNET_LIB_ACTIVATIONS_HPP
#define NEURALNET_LIB_ACTIVATIONS_HPP

#include "Dense"

using Eigen::MatrixXd;

namespace Activations { //TODO change inline

    class Activation {
    public:
        virtual inline std::string get_name() { return "abstract"; };
        virtual inline MatrixXd activate(const MatrixXd &input) { return MatrixXd(input); }
        virtual inline MatrixXd activate_back(const MatrixXd &dA, const MatrixXd &Z) { return MatrixXd(dA); }
    };

    class Linear: public Activation {
        inline std::string get_name() override { return "linear"; };
        inline MatrixXd activate(const MatrixXd &input) override {
            return MatrixXd(input);
        }
        inline MatrixXd activate_back(const MatrixXd &dA, const MatrixXd &Z) override {
            return MatrixXd(dA);
        }
    };

    class Sigmoid: public Activation {
        inline std::string get_name() override { return "sigmoid"; };
        inline MatrixXd activate(const MatrixXd &input) override {
            return ((-input.array()).exp() + 1).inverse();
        }
        inline MatrixXd activate_back(const MatrixXd &dA, const MatrixXd &Z) override {
            MatrixXd sigmoid_Z = activate(Z);
            return dA.array() * sigmoid_Z.array() * (1 - sigmoid_Z.array());
        }
    };


    class Softmax: public Activation {
        inline std::string get_name() override { return "softmax"; };
        inline MatrixXd activate(const MatrixXd &input) override {
            MatrixXd exp_input = input.array().exp();
            return exp_input / exp_input.sum();
        }
        inline MatrixXd activate_back(const MatrixXd &dA, const MatrixXd &Z) override {
            return MatrixXd(dA);
        }
    };

    class Tanh: public Activation {
        inline std::string get_name() override { return "tanh"; };
        inline MatrixXd activate(const MatrixXd &input) override {
            return input.array().tanh();
        }
        inline MatrixXd activate_back(const MatrixXd &dA, const MatrixXd &Z) override {
            return dA.array() / Z.array().cosh().square();
        }
    };

    class Relu: public Activation {
        inline std::string get_name() override { return "relu"; };
        inline MatrixXd activate(const MatrixXd &input) override {
            return input.array().max(0);
        }
        inline MatrixXd activate_back(const MatrixXd &dA, const MatrixXd &Z) override {
            return dA.array() * Z.unaryExpr([](double x) { return (double) (x > 0); }).array();
        }
    };
}

#endif //NEURALNET_LIB_ACTIVATIONS_HPP
