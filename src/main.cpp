#include <iostream>
#include <fstream>

#include "layer.h"
#include "losses.h"
#include "model.h"
#include "optimizers.h"



int main() {
    std::vector<Layers::Layer*> layers = {
            new Layers::Dense(20, 10, "tanh", "he"),
            new Layers::Dense(10, 5, "relu", "he"),
            new Layers::Dense(5, 1, "sigmoid", "he")
    };

    Model model(layers);


    model.compile(
            new Losses::BinaryCrossentropy(),
            new Optimizers::optimizers(0.01)
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


    MatrixXd train_prediction = model.predict(X);
//    MatrixXd test_prediction = model.predict(X_test_pts);

    std::cout << train_prediction << std::endl;

//    std::cout.precision(5);
//    std::cout << "Accuracy on the train set: " << compare(train_prediction, Y) * 100 << "%" << std::endl;
//    std::cout << "Accuracy on the test set: " << compare(test_prediction, Y) * 100 << "%" << std::endl;
//


//    std::cout << model.predict(X) << std::endl;
}
