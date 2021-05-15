//
// Created by bohdansydor on 15.05.21.
//



#include <Eigen/Core>
#include <iostream>
using namespace Eigen;
using namespace std;
int main(void)
{
    int const N = 2;
    MatrixXi A(N,N);
    A.setRandom();
    cout << "A =\n" << A << '\n' << endl;
//    auto a = A.middleCols(1,2);
//    a(0, 0) = 1000000000;
    A += A;
    cout << "A =\n" << A << endl;
    return 0;
}