//
// Created by raven on 5/14/21.
//

#include "layers.h"


void Layers::Conv2D::constructor(const std::vector<int> &input_shape, int n_filters, int kernel_size,
                         const std::vector<int> &strides, const std::string &padding,
                         const std::string &parameter_initialization) {

    if (input_shape.size() != 3 || strides.size() != 3) {
        throw std::logic_error("input_shape and strides must be of length 3");
    }
    if (padding != "valid" && padding != "same") {
        throw std::logic_error("padding must be 'same' or 'valid'");
    }

    this->kernel_size = kernel_size;
    this->strides = strides;

    this->n_C_prev = input_shape[2];
    this->n_C = n_filters;

    if (padding == "valid") { // (n + 2p - f // s) + 1
        this->n_H = input_shape[0] - kernel_size + 1; // TODO account for strides
        this->n_W = input_shape[1] - kernel_size + 1;
    } else if (padding == "same") {
        this->n_H = input_shape[0];
        this->n_W = input_shape[1];
    }

    // initializing filters
    std::default_random_engine gen{static_cast<long unsigned int>(time(nullptr))};
    double stddev;
    if (parameter_initialization == "normal") {
        stddev = 1;
    } else if (parameter_initialization == "kaiming" || parameter_initialization == "he") {
        stddev = std::sqrt(2. / (kernel_size * kernel_size * this->n_C_prev));
    } else if (parameter_initialization == "xavier" || parameter_initialization == "glorot") {
        stddev = std::sqrt(6. / (kernel_size * kernel_size * this->n_C_prev + n_filters));
    } else {
        throw std::logic_error("parameter initialization is not one of allowed");
    }

    std::normal_distribution<double> dist(0, stddev);
    this->W = Data3D(
            this->n_C, Data2D(
                    input_shape[2], MatrixXd::NullaryExpr( // no idea how this works
                            kernel_size, kernel_size, [&](){ return dist(gen); })));

    this->W = Data3D(
            this->n_C, Data2D(
                    1, MatrixXd::Zero(1, 1)));
}

Layers::Conv2D::Conv2D(const std::vector<int> &input_shape, int n_filters, int kernel_size, const std::vector<int> &strides,
               const std::string &padding, const std::string &parameter_initialization) {
    constructor(input_shape, n_filters, kernel_size, strides, padding, parameter_initialization);
}

Layers::Conv2D::Conv2D(const std::vector<int> &input_shape, int n_filters, int kernel_size, const std::vector<int> &strides,
               const std::string &padding) {
    constructor(input_shape, n_filters, kernel_size, strides, padding, "normal");
}

Layers::Conv2D::Conv2D(const std::vector<int> &input_shape, int n_filters, int kernel_size, const std::vector<int> &strides) {
    constructor(input_shape, n_filters, kernel_size, strides, "valid", "normal");
}

Layers::Conv2D::Conv2D(const std::vector<int> &input_shape, int n_filters, int kernel_size) {
    std::vector<int> strides(input_shape.size(), 1);
    constructor(input_shape, n_filters, kernel_size, strides, "valid", "normal");
}

double Layers::Conv2D::convolve(const Data2D &inp, int c, int h, int w) {
    // inefficient approach, will be refactored

    double result;

    for  (int c_prev = 0; c_prev < n_C_prev; ++c_prev) {
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                result += W[c][c_prev](i, j) * inp[c_prev](h + i, w + j);
            }
        }
    }

    result += b[c][0](0, 0); // single b for all entries in the filter;

    return result;
}

Data3D Layers::Conv2D::forward(const Data3D &inp, std::unordered_map<std::string, Data3D> &cache) {
    // TODO consider strides and same padding

    // Construct a container for resulting data
    int m = (int) inp.size(); // TODO create Data API for ease for interaction

    Data3D res(
            m, Data2D(
                    n_C, MatrixXd( // no idea how this works
                            n_H, n_W)));

    // TODO implement a much more effective approach.
    //  One source: https://medium.com/analytics-vidhya/implementing-convolution-without-for-loops-in-numpy-ce111322a7cd
    for (int i = 0; i < m; ++i) {
        for (int c = 0; c < n_C; ++c) {
            for (int h = 0; h < n_H; ++h) {
                for (int w = 0; w < n_W; ++w) {
                    res[i][c](h, w) = convolve(inp[i], c, h, w);
                }
            }
        }
    }

    return res;
}

Data3D Layers::Conv2D::backward(const Data3D &inp, std::unordered_map<std::string, Data3D> &cache) {
    return Data3D();
}

void Layers::Conv2D::update_parameters(std::unordered_map<std::string, Data3D> &cache) {}

std::unordered_map<std::string, std::vector<int>> Layers::Conv2D::layer_shapes() {
    return std::unordered_map<std::string, std::vector<int>>();
}
