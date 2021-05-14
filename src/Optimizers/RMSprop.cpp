//
// Created by bohdansydor on 13.05.21.
//
#include "optimizers.h"
#include "model.h"

Optimizers::RMSprop::RMSprop(int _batch_size,  double _learning_rate, double _beta, double _epsilon) {
    this->learning_rate = _learning_rate;
    this->beta = _beta;
    this->beta = _batch_size;
    this->epsilon = _epsilon; // to avoid division by zero
}

void Optimizers::RMSprop::optimize(Model *model, const MatrixXd &X_train, const MatrixXd &Y_train, int num_epochs) {
    std::vector<Layers::Layer *> layers = model->get_layers();

    auto rms_cache = *(new std::vector<std::unordered_map<std::string, MatrixXd>>(layers.size()));
    for (int l = 0; l < layers.size(); ++l) {
        std::unordered_map<std::string, std::vector<int>> layer_shapes = layers[l]->layer_shapes();
        for (const auto &grad: layers[l]->gradients) {
            rms_cache[l][grad] = MatrixXd::Zero(layer_shapes[grad][0], layer_shapes[grad][1]);
        }
    }

    bool stop = false;
    int i = 0;
    double prev_cost = 100;
    while (i < num_epochs && !stop) {
        int j = 0;
        while (j < X_train.cols() && !stop){
            MatrixXd Y_pred = model->forward(X_train.col(j));
//            std::cout << X_train.col(j) << std::endl;


            double cost = model->compute_cost(Y_pred, Y_train.col(j)); // can print or something;
            if ((i % (num_epochs / 10) == 0 || i == num_epochs - 1) && j==0) {
                std::cout << "Cost at iteration " << i << ": " << cost << std::endl;
            }

            model->backward(Y_pred, Y_train.col(j));

            std::cout << "\ndiff: " << std::abs(prev_cost - cost) << std::endl;
            std::cout << "prev_cost: " << prev_cost << std::endl;
            std::cout << "cost: " << cost << std::endl;
            if (std::abs(prev_cost - cost) < 0.1){
                stop = true;
                break;
            } else {
                prev_cost = cost;
            }

            update_parameters(model->get_layers(), model->get_cache(), rms_cache);
            j++;

        }
        i++;
    }
}



void Optimizers::RMSprop::update_parameters(
        std::vector<Layers::Layer *> &layers,
        std::vector<std::unordered_map<std::string, MatrixXd>> &cache,
        std::vector<std::unordered_map<std::string, MatrixXd>> &rms_cache)
{
    // this optimizer accounts uses the exponentially weighted average of squares of gradients
    for (int l = 0; l < layers.size(); ++l) {
        for (const auto &grad: layers[l]->gradients) {
            rms_cache[l][grad] = beta * rms_cache[l][grad] + (1 - beta) * cache[l][grad].array().square().matrix();
            cache[l][grad] = learning_rate * (cache[l][grad].array() / (rms_cache[l][grad].array() + epsilon).sqrt());
        }
        layers[l]->update_parameters(cache[l]);
    }
}

