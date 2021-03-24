#ifndef LINALG_CPP
#define LINALG_CPP

#include <ostream>
#include <iostream>
#include "linalg.h"

template<typename T>
Matrix<T>::Matrix(size_t rows_num, size_t cols_num, const T &_initial_val) {
    data.resize(rows_num);
    for (auto &r: data) { r.resize(cols_num, _initial_val); }
    rows = rows_num;
    cols = cols_num;
}

template<typename T>
Matrix<T>::Matrix(std::vector<std::vector<T> > &v) {
    size_t rows_num = v.size();
    size_t cols_num = v[0].size();

    data.resize(rows_num);
    for (auto &r: data) { r.resize(cols_num); }


    for (size_t i = 0; i < rows_num; i++) {
        for (size_t j = 0; j < cols_num; j++) {
            data[i][j] = v[i][j];
        }
    }
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
Matrix<T> Matrix<T>::operator*(const Matrix<T> &m2) {
//    when we multiplies two matrices smaller have to be rhs
    Matrix<T> res(this->get_rows(), this->get_rows(), 0.0);

    size_t m2_rows = m2.get_rows();
    size_t m2_cols = m2.get_cols();

    for (size_t i = 0; i < this->get_rows(); i++) {
        for (size_t j = 0; j < this->get_cols(); j++) {
            res(i, j) += data[i][j] * m2(i % m2_rows, j % m2_cols);
        }
    }
    return res;
}

template<typename T>
void Matrix<T>::print() {
    for (int i = 0; i < this->get_rows(); i++) {
        for (int j = 0; j < this->get_cols(); j++) {
            std::cout << data[i][j] << " ";
        }
        std::cout << "\n";
    }
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &m2) {
//    when we adding two matrices smaller have to be rhs
    Matrix<T> res(this->get_rows(), this->get_rows(), 0.0);

    size_t m2_rows = m2.get_rows();
    size_t m2_cols = m2.get_cols();

    for (size_t i = 0; i < this->get_rows(); i++) {
        for (size_t j = 0; j < this->get_cols(); j++) {
            res(i, j) += data[i][j] + m2(i % m2_rows, j % m2_cols);
        }
    }
    return res;
}

template<typename T>
Matrix<T>::Matrix() {
    rows = 0;
    cols = 0;
    std::vector<std::vector<T> > m;
    data = m;
}

template<typename T>
Matrix<T> transpose(Matrix<T> &m) {
    size_t row_num = m.get_rows();
    size_t col_num = m.get_cols();

    Matrix<T> res(col_num, row_num, 0.0);

    for (size_t i = 0; i < row_num; i++) {
        for (size_t j = 0; j < col_num; j++) {
            res(j, i) += m(i, j);
        }
    }
    return res;
}

template<typename T>
Matrix<T> dot(Matrix<T> &m1, Matrix<T> &m2) {
    if (m1.get_cols() != m2.get_rows()) { exit(1); }

    size_t new_rows_num = m1.get_rows();
    size_t new_cols_num = m2.get_cols();
    size_t m1_cols = m1.get_cols();
    Matrix<T> res(new_rows_num, new_cols_num, 0.0);

    for (size_t i = 0; i < new_rows_num; i++) {
        for (size_t j = 0; j < new_cols_num; j++) {
            for (size_t k = 0; k < m1_cols; k++) {
                res(i, j) += m1(i, k) * m2(k, j);
            }
        }
    }

    return res;
}

#endif