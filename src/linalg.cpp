#ifndef LINALG_CPP
#define LINALG_CPP

#include <ostream>
#include "linalg.h"

template<typename T>
Matrix<T>::Matrix(size_t rows_num, size_t cols_num, const T &_initial_val) {
    data.resize(rows_num);
    for (auto &r: data) { r.resize(cols_num, _initial_val); }
    rows = rows_num;
    cols = cols_num;
}

template<typename T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &e2) {
    if (&e2 == this) { return *this; }

    size_t new_rows_num = e2.get_rows();
    size_t new_cols_num = e2.get_cols();

    data.resize(e2.get_rows());
    for (auto &r: data) { r.resize(new_cols_num); }


    for (size_t r = 0; r < new_rows_num; r++) {
        data[r] = e2.data[r];
    }

    rows = new_rows_num;
    cols = new_cols_num;

    return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const T &e2) {
    Matrix res(rows, cols, 0.0);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            res(i, j) += data[i][j] + e2;
        }
    }

    return res;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const T &e2) {
    Matrix res(rows, cols, 0.0);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            res(i, j) += data[i][j] * e2;
        }
    }

    return res;
}

template<typename T>
size_t Matrix<T>::get_rows() const {
    return rows;
}

template<typename T>
size_t Matrix<T>::get_cols() const {
    return cols;
}


template<typename T>
T &Matrix<T>::operator()(const size_t &row, const size_t &col) {
    return data[row][col];
}

template<typename T>
const T &Matrix<T>::operator()(const size_t &row, const size_t &col) const {
    return data[row][col];
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &e2) {
    Matrix<T> res(rows, cols, 0.0);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            for (size_t k = 0; k < rows; k++) {
                res(i, j) += data[i][k] * e2(k, j);
            }
        }
    }

    return res;
}

template<typename T>
Matrix<T> transpose(Matrix<T> &m) {
    size_t row_num = m.get_rows();
    size_t col_num = m.get_cols();

    Matrix<T> res(row_num, col_num, 0.0);

    for (size_t i = 0; i < row_num; i++) {
        for (size_t j = 0; j < col_num; j++) {
            res(i, j) += m(j, i);
        }
    }
    return res;
}

#endif