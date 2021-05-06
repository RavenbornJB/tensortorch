//
// Created by raven on 4/30/21.
//

#ifndef NEURALNET_LIB_LAYER_H
#define NEURALNET_LIB_LAYER_H

#include <unordered_map>
#include <Eigen/Dense>

using Eigen::MatrixXd;


class Layer {
public:
    std::string name;
    virtual MatrixXd forward(const MatrixXd& inp, std::unordered_map<std::string, MatrixXd>& cache) { return inp; };
    virtual MatrixXd backward(const MatrixXd& inp, std::unordered_map<std::string, MatrixXd>& cache) { return inp; };
    virtual void update_parameters(std::unordered_map<std::string, double>& hparams, std::unordered_map<std::string, MatrixXd>& cache) {};
    virtual std::pair<std::vector<int>, std::vector<int>> get_shape() { return {{0}, {0}}; };
};


#endif //NEURALNET_LIB_LAYER_H
