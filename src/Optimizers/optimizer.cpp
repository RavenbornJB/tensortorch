//
// Created by bohdansydor on 13.05.21.
//

#include "optimizers.hpp"
#include "model.hpp"

void Optimizers::Optimizer::optimize(Model *model, const MatrixXd &X_train, const MatrixXd &Y_train, int num_epochs) {};

void Optimizers::Optimizer::update_parameters(std::vector<Layers::Layer *> &layers,
                                              std::vector<std::unordered_map<std::string, MatrixXd>> &cache,
                                              std::vector<std::unordered_map<std::string, MatrixXd>> &rms_cache) {};

