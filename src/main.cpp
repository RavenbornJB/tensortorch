#include <iostream>
#include "linalg.h"


template<typename T>
void printm(Matrix<T> a) {
    for (int i = 0; i < a.get_rows(); i++) {
        for (int j=0; j < a.get_cols(); j++) {
            std::cout << a(i, j) << " ";
        }
        std::cout << "\n";
    }
}


int main() {
    Matrix<int> a(2, 2, 1);
    a(0, 1) = 0;
    printm(a);
    printm(transpose(a));

    return 0;
}
