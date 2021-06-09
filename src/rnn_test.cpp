//
// Created by raven on 6/9/21.
//

#include <iostream>

#include "layer.hpp"
#include "activations.hpp"
#include "losses.hpp"

int main() {
    Layers::RNN layer(10, 16, 10, new Activations::Tanh,
                      new Activations::Softmax, "he", 5);

    MatrixXd x(10, 5);
    x.col(0) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    x.col(1) << 0, 0, 0, 0, 0, 0, 0, 0, 1, 0;
    x.col(2) << 0, 0, 0, 0, 1, 0, 0, 0, 0, 0;
    x.col(3) << 0, 0, 0, 0, 0, 0, 0, 1, 0, 0;
    x.col(4) << 0, 0, 0, 0, 0, 0, 1, 0, 0, 0;

    MatrixXd y_true(10, 5);
    MatrixXd eos_token(10, 1);
    eos_token << 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    y_true << x.block<10, 4>(0, 1), eos_token;

    std::unordered_map<std::string, MatrixXd> cache;
    Losses::CategoricalCrossentropy loss_function;

    int num_epochs = 10000;
    MatrixXd y_pred;
    MatrixXd dA;
    for (int i = 0; i < num_epochs; ++i) {
        y_pred = layer.forward(x, cache);

        if (i % (num_epochs / 10) == 0) {
            std::cout << "Loss at iteration " << i << ": " << loss_function.loss(y_pred, y_true).mean() << std::endl;
        }

        dA = loss_function.loss_back(y_pred, y_true); // handled by model

        layer.backward(dA, cache); // previous layer needs dX, it's their start of backprop, but here don't need it

        layer.update_parameters(cache);
    }

    std::cout << y_pred << std::endl;
}
