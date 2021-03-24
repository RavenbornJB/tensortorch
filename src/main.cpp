#include <iostream>
#include "linalg.h"

template<typename T>
void printm(Matrix<T> m) {
    for (int i = 0; i < m.get_rows(); i++) {
        for (int j = 0; j < m.get_cols(); j++) {
            std::cout << m(i, j) << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    std::vector<int> r11{1, 2, 3};
    std::vector<int> r12{4, 5, 6};
    std::vector<std::vector<int> > m1_v{r11, r12};
//
//    std::vector<int> c11{7, 8};
//    std::vector<int> c12{9, 10};
//    std::vector<int> c13{11, 12};
//    std::vector<std::vector<int> > m2_v{c11, c12, c13};
//
//    Matrix<int> a(m1_v);
//    Matrix<int> b(m2_v);
//
//    a.print();
//    std::cout << "\n" << std::endl;
////    printm(b);
////    std::cout << "\n" << std::endl;
////    printm(dot(a, b));
////    std::cout << "\n" << std::endl;
//    transpose(a).print();

    Matrix<int> a(4, 4, 1);
    Matrix<int> b(4, 1, 1);
    Matrix<int> c;

    c = dot(a, b);

    transpose(c);

    return 0;
}
