//
// Created by bohdan on 11.06.21.
//

#ifndef TENSORTORCHAF_LOSSES_HPP
#define TENSORTORCHAF_LOSSES_HPP

#include <arrayfire.h>

using af::array;

namespace Losses {

    class Loss {
    public:
        std::string name;

        virtual inline array loss(const array &y_pred, const array &y_true) { return y_pred; }

        virtual inline array loss_back(const array &y_pred, const array &y_true) { return y_pred; }
    };


    class BinaryCrossentropy : public Loss {
    public:
        std::string name = "binary_crossentropy";

        inline array loss(const array &y_pred, const array &y_true) override {
            array logProb = y_true * af::log(y_pred) + (1 - y_true) * af::log(1 - y_pred);
            return -logProb;
        }

        inline array loss_back(const array &y_pred, const array &y_true) override {
            array logProb = y_true / y_pred - (1 - y_true) / (1 - y_pred);
            return -logProb;
        }
    };
}

#endif //TENSORTORCHAF_LOSSES_HPP
