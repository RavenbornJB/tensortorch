//
// Created by raven on 3/24/21.
//

#include "utility.h"


Matrix<double> sum(const Matrix<double> &m, size_t axis) {
    Matrix<double> output;
    size_t rows = m.get_rows(), cols = m.get_cols();
    if (axis == 0) {
        output = *new Matrix<double>(1, cols, 0);
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) output(0, j) += m(i, j);
        }
    } else if (axis == 1) {
        output = *new Matrix<double>(rows, 1, 0);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) output(i, 0) += m(i, j);
        }
    } else {
        throw std::logic_error("Invalid axis value");
    }
    return output;
}

double sum(const Matrix<double> &m) {
    double res = 0;
    size_t rows = m.get_rows(), cols = m.get_cols();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) res += m(i, j);
    }
    return res;
}
