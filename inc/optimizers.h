//
// Created by raven on 5/11/21.
//

// All formulae for advanced optimization algorithms in this file come from
// TODO insert article link, I know which

#ifndef NEURALNET_LIB_OPTIMIZERS_H
#define NEURALNET_LIB_OPTIMIZERS_H

#include <iostream>

#include "layer.h"


namespace Optimizers {
    class Optimizer {
    public:
        virtual void update_parameters(
                std::vector<Layer*> &layers, std::vector<std::unordered_map<std::string, MatrixXd>>& cache
                ) {}
    };

    class Default: public Optimizer {
    private:
        double learning_rate;

    public:
        Default(std::vector<Layer*> &layers, double learning_rate) {
            this->learning_rate = learning_rate;
        }

        void update_parameters(
                std::vector<Layer*> &layers, std::vector<std::unordered_map<std::string, MatrixXd>>& cache
        ) override {
            // basic optimizer, just uses the learning rate
            for (int l = 0; l < layers.size(); ++l) {
                for (const auto& grad: layers[l]->gradients) {
                    cache[l][grad] *= learning_rate;
                }
                layers[l]->update_parameters(cache[l]);
            }
        }
    };

    class SGD: public Optimizer { // TODO test on normal data, others too
    private:
        double learning_rate;
        double momentum;
        std::vector<std::unordered_map<std::string, MatrixXd>> momentum_cache;

    public:
        SGD(std::vector<Layer *> &layers, double learning_rate, double momentum) {
            this->learning_rate = learning_rate;
            this->momentum = momentum;
            momentum_cache = *(new std::vector<std::unordered_map<std::string, MatrixXd>>(layers.size()));

            for (int l = 0; l < layers.size(); ++l) {
                std::unordered_map<std::string, std::vector<int>> layer_shapes = layers[l]->layer_shapes();
                for (const auto &grad: layers[l]->gradients) {
                    this->momentum_cache[l][grad] = MatrixXd::Zero(layer_shapes[grad][0], layer_shapes[grad][1]);
                }
            }
        }

        void update_parameters(
                std::vector<Layer *> &layers, std::vector<std::unordered_map<std::string, MatrixXd>> &cache
        ) override {
            // this optimizer accounts uses the exponentially weighted average of gradients
            for (int l = 0; l < layers.size(); ++l) {
                for (const auto &grad: layers[l]->gradients) {
                    momentum_cache[l][grad] = momentum * momentum_cache[l][grad] + (1 - momentum) * cache[l][grad];
                    cache[l][grad] = momentum_cache[l][grad] * learning_rate;
                }
                layers[l]->update_parameters(cache[l]);
            }
        }
    };

    class RMSprop: public Optimizer { // TODO i'm not even sure this works
    private:
        double learning_rate;
        double beta;
        double epsilon;
        std::vector<std::unordered_map<std::string, MatrixXd>> rms_cache;

    public:
        RMSprop(std::vector<Layer *> &layers, double learning_rate, double beta) {
            this->learning_rate = learning_rate;
            this->beta = beta;
            this->epsilon = std::pow(10, -7); // to avoid division by zero
            rms_cache = *(new std::vector<std::unordered_map<std::string, MatrixXd>>(layers.size()));

            for (int l = 0; l < layers.size(); ++l) {
                std::unordered_map<std::string, std::vector<int>> layer_shapes = layers[l]->layer_shapes();
                for (const auto &grad: layers[l]->gradients) {
                    this->rms_cache[l][grad] = MatrixXd::Zero(layer_shapes[grad][0], layer_shapes[grad][1]);
                }
            }
        }

        void update_parameters(
                std::vector<Layer *> &layers, std::vector<std::unordered_map<std::string, MatrixXd>> &cache
        ) override {
            // this optimizer accounts uses the exponentially weighted average of squares of gradients
            for (int l = 0; l < layers.size(); ++l) {
                for (const auto &grad: layers[l]->gradients) {
                    rms_cache[l][grad] = beta * rms_cache[l][grad] + (1 - beta) * cache[l][grad].array().square().matrix();
                    cache[l][grad] = learning_rate * (cache[l][grad].array() / (rms_cache[l][grad].array() + epsilon).sqrt());
                }
                layers[l]->update_parameters(cache[l]);
            }
        }
    };

    class Adam: public Optimizer { // TODO Adam, for now copy of SGD
    private:
        double learning_rate;
        double momentum;
        std::vector<std::unordered_map<std::string, MatrixXd>> momentum_cache;

    public:
        Adam(std::vector<Layer *> &layers, double learning_rate, double momentum) {
            this->learning_rate = learning_rate;
            this->momentum = momentum;
            momentum_cache = *(new std::vector<std::unordered_map<std::string, MatrixXd>>(layers.size()));

            for (int l = 0; l < layers.size(); ++l) {
                std::unordered_map<std::string, std::vector<int>> layer_shapes = layers[l]->layer_shapes();
                for (const auto &grad: layers[l]->gradients) {
                    this->momentum_cache[l][grad] = MatrixXd::Zero(layer_shapes[grad][0], layer_shapes[grad][1]);
                }
            }
        }

        void update_parameters(
                std::vector<Layer *> &layers, std::vector<std::unordered_map<std::string, MatrixXd>> &cache
        ) override {
            // this optimizer accounts uses the exponentially weighted average
            for (int l = 0; l < layers.size(); ++l) {
                for (const auto &grad: layers[l]->gradients) {
                    momentum_cache[l][grad] = momentum * momentum_cache[l][grad] + (1 - momentum) * cache[l][grad];
                    cache[l][grad] = momentum_cache[l][grad] * learning_rate;
                }
                layers[l]->update_parameters(cache[l]);
            }
        }
    };
}

#endif //NEURALNET_LIB_OPTIMIZERS_H
