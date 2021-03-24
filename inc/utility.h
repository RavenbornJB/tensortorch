//
// Created by raven on 3/24/21.
//

#ifndef NEURALNET_LIB_UTILITY_H
#define NEURALNET_LIB_UTILITY_H

#include <stdexcept>

#include "linalg.h"


Matrix<double> sum(const Matrix<double> &m, size_t axis);
double sum(const Matrix<double> &m);

#endif //NEURALNET_LIB_UTILITY_H
