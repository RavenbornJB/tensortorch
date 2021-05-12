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
    std::vector<std::string> description;
    std::vector<std::string> gradients;
    virtual MatrixXd forward(const MatrixXd& inp, std::unordered_map<std::string, MatrixXd>& cache) { return inp; };
    virtual MatrixXd backward(const MatrixXd& inp, std::unordered_map<std::string, MatrixXd>& cache) { return inp; };
    virtual void update_parameters(std::unordered_map<std::string, MatrixXd>& cache) {};
    // layer_shapes uses "d" in front of keys like in gradients, because this is used in the optimizer for gradients
    virtual std::unordered_map<std::string, std::vector<int>> layer_shapes() {};
};


#endif //NEURALNET_LIB_LAYER_H
