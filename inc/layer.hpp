//
// Created by raven on 4/30/21.
//

#ifndef NEURALNET_LIB_LAYER_HPP
#define NEURALNET_LIB_LAYER_HPP

#include <unordered_map>
#include <random>
#include <iostream>
#include <string>
#include <cmath>

#include "Dense"
#include "activations.hpp"

using Eigen::MatrixXd;

namespace Layers {

    class Layer {
    public:
        std::vector<std::string> description;
        std::vector<std::string> gradients;

        virtual MatrixXd forward(const MatrixXd &inp, std::unordered_map<std::string, MatrixXd> &cache) { return inp; };

        virtual MatrixXd
        backward(const MatrixXd &inp, std::unordered_map<std::string, MatrixXd> &cache) { return inp; };

        virtual void update_parameters(std::unordered_map<std::string, MatrixXd> &cache) {};

        // layer_shapes uses "d" in front of keys like in gradients, because this is used in the optimizer for gradients
        virtual std::unordered_map<std::string, std::vector<int>> layer_shapes() {};
    };

    class Dense: public Layer {
    private:
        MatrixXd W;
        MatrixXd b;

        void constructor(int from_size, int to_size, Activations::Activation* activation, const std::string &parameter_initialization);

        Activations::Activation* activation;

        MatrixXd linear(const MatrixXd& input);
        MatrixXd linear_backward(const MatrixXd& dZ, std::unordered_map<std::string, MatrixXd>& cache);

    public:
        Dense(int from_size, int to_size, const std::string &activation_type, const std::string &parameter_initialization);
        Dense(int from_size, int to_size, const std::string &activation_type);
        Dense(int from_size, int to_size, Activations::Activation* activation, const std::string &parameter_initialization);
        Dense(int from_size, int to_size, Activations::Activation* activation);
        Dense(int from_size, int to_size);
        MatrixXd forward(const MatrixXd& input, std::unordered_map<std::string, MatrixXd>& cache) override;
        MatrixXd backward(const MatrixXd& dA, std::unordered_map<std::string, MatrixXd>& cache) override;
        void update_parameters(std::unordered_map<std::string, MatrixXd>& cache) override;
        std::unordered_map<std::string, std::vector<int>> layer_shapes() override;
    };

    class RNN: public Layer {
    private:
        MatrixXd Waa;
        MatrixXd Wax;
        MatrixXd Wya;
        MatrixXd ba;
        MatrixXd by;

        int input_size;
        int hidden_size;
        int output_size;
        int sequence_size;

        Activations::Activation* activation_a;
        Activations::Activation* activation_y;

        MatrixXd cell_forward(const MatrixXd& input, MatrixXd& hidden, const std::string& timestep,
                              std::unordered_map<std::string, MatrixXd>& cache);

        MatrixXd linear_backward_y(const MatrixXd &dZ, const std::string& timestep,
                                 std::unordered_map<std::string, MatrixXd>& cache);
        MatrixXd cell_backward_y(const MatrixXd &dA, const std::string& timestep,
                                   std::unordered_map<std::string, MatrixXd>& cache);
        std::pair<MatrixXd, MatrixXd> cell_backward_a(const MatrixXd& dA, const std::string& timestep,
                                                      std::unordered_map<std::string, MatrixXd>& cache);
        std::pair<MatrixXd, MatrixXd> cell_backward(const MatrixXd& dA, const MatrixXd& dA_next,
                                                    const std::string& timestep,
                                                    std::unordered_map<std::string, MatrixXd>& cache);

        void initialize_gradient_caches(std::unordered_map<std::string, MatrixXd>& cache);
        bool initialized;

    public:
        RNN(int input_size, int hidden_size, int output_size, Activations::Activation *activation_class_a,
            Activations::Activation *activation_class_y, const std::string &parameter_initialization, int sequence_size);

        MatrixXd forward(const MatrixXd& input, std::unordered_map<std::string, MatrixXd>& cache) override;
        MatrixXd backward(const MatrixXd& dA, std::unordered_map<std::string, MatrixXd>& cache) override;
        void update_parameters(std::unordered_map<std::string, MatrixXd>& cache) override;
        std::unordered_map<std::string, std::vector<int>> layer_shapes() override;
    };

}
#endif //NEURALNET_LIB_LAYER_HPP
