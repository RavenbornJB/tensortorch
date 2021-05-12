#include <iostream>

#include "layers.h"
#include "losses.h"
#include "optimizers.h"
#include "model.h"


int main() {
    std::vector<Layer*> layers = {
            new Dense(20, 10, "tanh", "he"),
            new Dense(10, 5, "relu", "he"),
            new Dense(5, 1, "sigmoid", "he")
    };

    Model model(
            layers,
            new Losses::BinaryCrossentropy,
            new Optimizers::RMSprop(layers, 0.01, 0.999)
            );

    MatrixXd X(20, 2);
    MatrixXd Y(1, 2);

    for (int i = 0; i < 20; ++i) {
        X(i, 0) = 0.2;
        X(i, 1) = 0.8;
    }
    Y(0, 0) = 1;
    Y(0, 1) = 0;

    model.fit(X, Y, 1000);

    std::cout << model.predict(X) << std::endl;
}
