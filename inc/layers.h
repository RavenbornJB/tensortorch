//
// Created by raven on 4/30/21.
//

#ifndef NEURALNET_LIB_LAYER_H
#define NEURALNET_LIB_LAYER_H

#include <unordered_map>
#include <random>
#include <iostream>

#include "Dense"
#include "layers.h"
#include "activations.h"
#include "data.h"

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
        Activations::Activation* make_activation(const std::string& activation_type);

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

    class Conv2D { // TODO make conform to Layer
    private:
        Data3D W;
        Data3D b;

        int kernel_size;
        std::vector<int> strides;

        int n_C_prev;
        int n_C;
        int n_H;
        int n_W;
//    Activations::Activation* activation;

        void constructor(const std::vector<int>& input_shape, int n_filters, int kernel_size,
                         const std::vector<int>& strides, const std::string& padding,
                         const std::string &parameter_initialization);

        double convolve(const Data2D& inp, int c, int h, int w);
    public:
        std::vector<std::string> description;
        std::vector<std::string> gradients;

        Conv2D(const std::vector<int>& input_shape, int n_filters, int kernel_size,
               const std::vector<int>& strides, const std::string& padding,
               const std::string &parameter_initialization);
        Conv2D(const std::vector<int>& input_shape, int n_filters, int kernel_size,
               const std::vector<int>& strides, const std::string& padding);
        Conv2D(const std::vector<int>& input_shape, int n_filters, int kernel_size,
               const std::vector<int>& strides);
        Conv2D(const std::vector<int>& input_shape, int n_filters, int kernel_size);

        Data3D forward(const Data3D& inp, std::unordered_map<std::string, Data3D>& cache);
        Data3D backward(const Data3D& inp, std::unordered_map<std::string, Data3D>& cache);
        void update_parameters(std::unordered_map<std::string, Data3D>& cache);
        // layer_shapes uses "d" in front of keys like in gradients, because this is used in the optimizer for gradients
        virtual std::unordered_map<std::string, std::vector<int>> layer_shapes();
    };

}
#endif //NEURALNET_LIB_LAYER_H
