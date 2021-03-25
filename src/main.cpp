#include <iostream>

#include "linalg.h"
#include "layer.h"
#include "model.h"


int main() {
    def_layers_vector lp = {{3, "relu"}, {1, "sigmoid"}};
    Model shallow_model(2, lp, 0.01);
    std::vector<std::vector<double>> X_and_train_v = {{0., 0., 1., 1.}, {0., 1., 0., 1.}};
    std::vector<std::vector<double>> Y_and_train_v = {{0., 0., 0., 1.}};
    std::vector<std::vector<double>> X_and_test_v = {{1.}, {0.}};
    std::vector<std::vector<double>> Y_and_test_v = {{0.}};
    mdb X_and_train(X_and_train_v);
    mdb Y_and_train(Y_and_train_v);
    mdb X_and_test(X_and_test_v);
    mdb Y_and_test(Y_and_test_v);
    shallow_model.fit(X_and_train, Y_and_train, 1000);
    mdb res = shallow_model.predict(X_and_test);
    std::cout << (res(0, 0) == Y_and_test(0, 0)) << std::endl; // TODO make == in Matrix
}