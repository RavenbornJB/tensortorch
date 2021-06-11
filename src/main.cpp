#include <iostream>
#include <arrayfire.h>
#include <stdio.h>
#include <af/util.h>
#include <vector>
#include <fstream>
#include "activations.hpp"
#include "layer.hpp"
#include "losses.hpp"
#include "model.hpp"
#include "optimizers.hpp"

array get_data(const std::string &filename) {

    std::ifstream datafile(filename);
    if (!datafile.is_open()) throw std::invalid_argument("Incorrect file name");
    int num_points;
    datafile >> num_points;

//    std::cout << "num_points = " << num_points << std::endl;

    af::array data(3, 2000);
    double x, y, labele;

    for (size_t i = 0; i < num_points; ++i) {
        datafile >> x >> y >> labele;
        data(0, i) = x;
        data(1, i) = y;
        data(2, i) = labele;
    }

    return data;
}

//double compare(const array &prediction, const array &Y) {
//    size_t m = Y.dims(1);
//    double coincide = 0.;
//    for (size_t i = 0; i < m; ++i) {
//        coincide += ((prediction(0, i) >= 0.5) == Y(0, i));
//    }
//    return coincide / m;
//}

int main() {
    printf("Trying OpenCL Backend\n");
    af::setBackend(AF_BACKEND_OPENCL);
    af::info();

    af::array data = get_data("../data/shuffled_data_noisy_4_point.txt");

    af::array X_train_pts = data.rows(0, 1);
    af::array Y_train_pts = data.row(2);

    af_print(X_train_pts);
    af_print(Y_train_pts);

    std::vector<Layers::Layer *> layers = {
            new Layers::Dense(2, 5, new Activations::Relu, "he"),
//            new Layers::Dense(500, 5000, new Activations::Relu, "he"),
            new Layers::Dense(5, 1, new Activations::Sigmoid, "he")
    };

    Model model(layers);

    model.compile(
            new Losses::BinaryCrossentropy(),
            new Optimizers::BGD(0.0001)
    );



//    std::cout << "HERE" << std::endl;


    model.fit(X_train_pts, Y_train_pts, 100);

    array train_prediction = model.predict(X_train_pts);

    af_print(  af::sum((train_prediction >= 0.5) == Y_train_pts));
//    af_print(train_prediction);

//    std::cout << "Accuracy on the train set: " << compare(train_prediction, Y_train_pts) * 100 << "%" << std::endl;



}
