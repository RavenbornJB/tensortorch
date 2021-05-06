#include <iostream>

#include "layers.h"
#include "losses.h"
#include "model.h"


int main() {
    std::vector<Layer*> layers = {
            new Dense(20, 10, "tanh", "he"),
            new Dense(10, 5, "relu", "xavier"),
            new Dense(5, 3, "softmax")
    };

    Model model(layers, new Losses::CategoricalCrossentropy, "RMSprop", 0.01);

    MatrixXd input = MatrixXd::Constant(20, 1, 0.2);

//    auto res = model.forward(input);

//    std::cout << res << std::endl;
}
