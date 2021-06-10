#include <iostream>
#include <fstream>

#include "layer.hpp"
#include "losses.hpp"
#include "model.hpp"
#include "optimizers.hpp"

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
    double x, y, labele;
    for (size_t i = 0; i < num_points; ++i) {
//        if (i >= 5) {
//            continue;
//        }
        datafile >> x >> y;
        data.first[0].push_back(x);
        data.first[1].push_back(y);
        data.second[0].push_back(0);
    }

    return data;
}


datatype get_dat_shuffled(const std::string &filename) {
    std::ifstream datafile(filename);
    if (!datafile.is_open()) throw std::invalid_argument("Incorrect file name");
    size_t num_points;
    datafile >> num_points;

    datatype data;
    data.first.emplace_back();
    data.first.emplace_back();
    data.second.emplace_back();
    double x, y, label;
    for (size_t i = 0; i < num_points; ++i) {

        datafile >> x >> y >> label;
        data.first[0].push_back(x);
        data.first[1].push_back(y);
        data.second[0].push_back(label);
    }
//    for (size_t i = 0; i < num_points; ++i) {
////        if (i >= 5) {
////            continue;
////        }
//        datafile >> x >> y;
//        data.first[0].push_back(x);
//        data.first[1].push_back(y);
//        data.second[0].push_back(1);
//    }
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

int prediction_to_index(const Eigen::VectorXd& input, double threshold) {
    for (int r = 0; r < input.size(); ++r) {
        if (input(r) > threshold) return r;
    }
    return -1;
}

Eigen::VectorXd prediction_matrix_to_vector(const MatrixXd& input, double threshold) {
    Eigen::VectorXd res(input.cols());
    for (int c = 0; c < input.cols(); ++c) {
        res(c) = prediction_to_index(input.col(c), threshold);
    }
    return res;
}

int main() {
//
//    auto train_data = get_data("../data_generation/shuffled_data_noisy_4_point.txt");
//
//    MatrixXd X_train_pts = matrix_from_vector2d(train_data.first);
//    MatrixXd Y_train_pts = matrix_from_vector2d(train_data.second);
//    auto test_data = get_data("../data_generation/data_linear_test.txt");
//    MatrixXd X_test_pts = matrix_from_vector2d(test_data.first);
//    MatrixXd Y_test_pts = matrix_from_vector2d(test_data.second);
//
//
//    std::vector<Layers::Layer *> layers = {
//            new Layers::Dense(2, 5, "tanh", "he"),
//            new Layers::Dense(5, 5, "relu", "he"),
//            new Layers::Dense(5, 1, "sigmoid", "he")
//    };
//
//    Model model(layers);
//
//
//    model.compile(
//            new Losses::BinaryCrossentropy(),
////            new Optimizers::BGD(0.01)
////            new Optimizers::SGD(128, 0.01, 0.999)
////with higher learning rate gradient vanishing
////            new Optimizers::RMSprop(64, 0.00001, 0.999)
//            new Optimizers::Parallel(128, 0.1)
//    );
//
//    //TODO fix problem with number of epochs( <10)
//    model.fit(X_train_pts, Y_train_pts, 5000);
//
//    std::cout << "cols: " << Y_train_pts.cols() << "\nrows: " << Y_train_pts.rows() << std::endl;
//
//    MatrixXd train_prediction = model.predict(X_train_pts);
//    std::cout << "Accuracy on the train set: " << compare(train_prediction, Y_train_pts) * 100 << "%" << std::endl;

    MatrixXd x(20, 4);
    x.col(0) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0;
    x.col(1) << 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    x.col(2) << 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    x.col(3) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    MatrixXd y_true(20, 4);
    MatrixXd empty_token = MatrixXd::Zero(20, 1);
    y_true << x.block<20, 3>(0, 1), empty_token;

    std::vector<Layers::Layer *> layers = {
            new Layers::RNN(10, 16, 10, true,
            new Activations::Tanh,new Activations::Softmax, "he")
    };

    Model model(layers);
    model.compile(new Losses::CategoricalCrossentropy,
                  new Optimizers::Adam(1, 0.01, 0.9, 0.999));

    model.fit(x, y_true, 10000);

    auto res = model.predict(x.block(0, 0, 10, 4));
    std::cout << res << std::endl;
    std::cout << prediction_matrix_to_vector(res, 0.5) << std::endl;
}
