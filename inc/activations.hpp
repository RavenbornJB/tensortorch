#ifndef NEURALNET_LIB_ACTIVATIONS_HPP
#define NEURALNET_LIB_ACTIVATIONS_HPP

#include <arrayfire.h>

using af::array;

namespace Activations {

    class Activation {
    public:
        std::string name;
        virtual inline array activate(const array &input) { return input; }
        virtual inline array activate_back(const array &dA, const array &Z) { return dA; }
    };


    class Sigmoid: public Activation {
        std::string name = "sigmoid";
        inline array activate(const array &input) override {
            return 1/(1 + af::exp(-input));
        }
        inline array activate_back(const array &dA, const array &Z) override {
            array sigmoid_Z = activate(Z);
            return dA * sigmoid_Z * (1 - sigmoid_Z);
        }
    };


    class Relu: public Activation {
        std::string name = "relu";
        inline array activate(const array &input) override {
            return input * (input > 0);
        }
        inline array activate_back(const array &dA, const array &Z) override {
            return dA * (Z > 0);
        }
    };

}

#endif //NEURALNET_LIB_ACTIVATIONS_HPP
