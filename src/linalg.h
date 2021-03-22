#ifndef NN_PROJECT_LINALG_H
#define NN_PROJECT_LINALG_H

#include <vector>

template<typename T>
class Matrix {
private:
    size_t rows;
    size_t cols;
    std::vector<std::vector<T> > data;

public:

    Matrix(size_t _rows, size_t _cols, const T &_initial_val);

    Matrix<T> &operator=(const Matrix<T> &e2);

    Matrix<T> operator+(const T &e2);

    Matrix<T> operator*(const T &e2);

    Matrix<T> operator*(const Matrix<T> &rhs);

    size_t get_rows() const;

    size_t get_cols() const;

    T &operator()(const size_t &row, const size_t &col);

    const T &operator()(const size_t &row, const size_t &col) const;
};


#include "linalg.cpp"

#endif
