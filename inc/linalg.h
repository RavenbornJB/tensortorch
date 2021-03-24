#ifndef NN_PROJECT_LINALG_H
#define NN_PROJECT_LINALG_H

#include <vector>
#include <thread>
#include <mutex>
#include <iostream>

#define THREADS_NUM 4

struct matrix_ranges {
    size_t r1;
    size_t r2;
    size_t c1;
    size_t c2;
};

template<typename T>
class Matrix {
private:
    size_t rows{};
    size_t cols{};
    std::vector<std::vector<T> > data;

    void mlt_thr_add_scalar(matrix_ranges &r, Matrix<T> &res_m, T scalar, std::mutex &m);

public:
    Matrix();

    Matrix(size_t _rows, size_t _cols, const T &_initial_val);

    explicit Matrix(std::vector<std::vector<T> > &v);

    Matrix<T> &operator=(const Matrix<T> &e2);

    // matrix|scalar operations
    Matrix<T> operator+(const T &e2) const;

    Matrix<T> operator*(const T &e2) const;

    Matrix<T> operator-(const T &e2) const;

    Matrix<T> operator/(const T &e2) const;

    void operator+=(const T &e2);

    void operator*=(const T &e2);

    void operator-=(const T &e2);

    void operator/=(const T &e2);

    // matrix|matrix operations
    Matrix<T> operator+(const Matrix<T> &e2) const;

    Matrix<T> operator*(const Matrix<T> &e2) const;

    Matrix<T> operator-(const Matrix<T> &e2) const;

    Matrix<T> operator/(const Matrix<T> &e2) const;

    void operator+=(const Matrix<T> &e2);

    void operator*=(const Matrix<T> &e2);

    void operator-=(const Matrix<T> &e2);

    void operator/=(const Matrix<T> &e2);

    // access operators
    [[nodiscard]] size_t get_rows() const;

    [[nodiscard]] size_t get_cols() const;

    T &operator()(const size_t &row, const size_t &col);

    const T &operator()(const size_t &row, const size_t &col) const;

    void print() const;

    Matrix<T> sum(int8_t axis) const;

    T sum() const;

    void apply(T (*f)(T));
};


#include "linalg.cpp"

#endif