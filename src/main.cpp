#include <iostream>
#include <fstream>

#include "layer.h"
#include "losses.h"
#include "model.h"
#include "optimizers.h"

typedef
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
        datatype;

datatype get_data(const std::string &filename) {
    std::ifstream datafile(filename);
    if (!datafile.is_open()) throw std::invalid_argument("Incorrect file name");
    size_t num_points;
    datafile >> num_points;

    datatype data;
    data.first.emplace_back();
    data.first.emplace_back();
    data.second.emplace_back();
    double x, y;
    for (size_t i = 0; i < num_points; ++i) {
//        if (i >= 5) {
//            continue;
//        }
        datafile >> x >> y;
        data.first[0].push_back(x);
        data.first[1].push_back(y);
        data.second[0].push_back(0);
    }
    for (size_t i = 0; i < num_points; ++i) {
//        if (i >= 5) {
//            continue;
//        }
        datafile >> x >> y;
        data.first[0].push_back(x);
        data.first[1].push_back(y);
        data.second[0].push_back(1);
    }
    return data;
}


double compare(const MatrixXd &prediction, const MatrixXd &Y) {
    size_t m = Y.cols();
    double coincide = 0.;
    for (size_t i = 0; i < m; ++i) {
        coincide += (double) ((prediction(0, i) >= 0.5) == Y(0, i));
    }
    return coincide / m;
}

MatrixXd matrix_from_vector2d(std::vector<std::vector<double> > &v) {
    MatrixXd mat(v.size(), v[0].size());
    for (int i = 0; i < mat.rows(); i++)
        mat.row(i) = Eigen::VectorXd::Map(&v[i][0], v[i].size());
    return mat;
}

int main() {

    auto train_data = get_data("../data_generation/data_linear.txt");

    MatrixXd X_train_pts = matrix_from_vector2d(train_data.first);
    MatrixXd Y_train_pts = matrix_from_vector2d(train_data.second);
    auto test_data = get_data("../data_generation/data_linear_test.txt");
    MatrixXd X_test_pts = matrix_from_vector2d(test_data.first);
    MatrixXd Y_test_pts = matrix_from_vector2d(test_data.second);



    std::vector<Layers::Layer*> layers = {
            new Layers::Dense(2, 5, "tanh", "he"),
//            new Layers::Dense(10, 5, "relu", "he"),
            new Layers::Dense(5, 1, "sigmoid", "he")
    };

    Model model(layers);


    model.compile(
            new Losses::BinaryCrossentropy(),
//            new Optimizers::BGD(0.01)
//            new Optimizers::SGD(64, 0.01, 0.999)
            new Optimizers::RMSprop(64, 0.01, 0.999)
);


//    MatrixXd X(3, 10);
//
//    for (int i = 0; i < 10; ++i) {
//        for (int j=0; j < 3; ++j) {
//            X(j, i) = i;
//        }
//    }
//    MatrixXd Y(1, 2);
//    std::cout << X.middleCols(3, 3) << std::endl;

//    Y(0, 0) = 1;
//    Y(0, 1) = 0;


    //TODO fix problem with number of epochs( <10)
    model.fit(X_train_pts, Y_train_pts, 10);


    MatrixXd train_prediction = model.predict(X_train_pts);
//    MatrixXd test_prediction = model.predict(X_test_pts);

//    std::cout << train_prediction << std::endl;
//    std::cout << Y_train_pts << std::endl;
//    std::cout.precision(5);
    std::cout << "Accuracy on the train set: " << compare(train_prediction, Y_train_pts) * 100 << "%" << std::endl;
//    std::cout << "Accuracy on the test set: " << compare(test_prediction, Y) * 100 << "%" << std::endl;
//


//    std::cout << model.predict(X) << std::endl;
}
