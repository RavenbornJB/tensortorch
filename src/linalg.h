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

    Matrix(unsigned _rows, unsigned _cols, const T &_initial_val);

    // Operator overloading, for "standard" mathematical matrix operations
    Matrix<T> &operator=(const Matrix<T> &e2);

    Matrix<T> transpose();

    // Matrix/scalar operations
    Matrix<T> operator+(const T &e2);

    Matrix<T> operator*(const T &e2);

    // Matrix/vector operations
    Matrix<T> operator*(const Matrix<T> &rhs);


    // Access the row and column sizes
    size_t get_rows() const;

    size_t get_cols() const;

    T get(size_t i, size_t j) const;

    T &operator()(const unsigned &row, const unsigned &col);

    const T &operator()(const unsigned &row, const unsigned &col) const;
};

#include "linalg.cpp"

#endif
