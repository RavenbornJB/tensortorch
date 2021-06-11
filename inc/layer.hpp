//
// Created by bohdan on 11.06.21.
//

#ifndef TENSORTORCHAF_LAYER_HPP
#define TENSORTORCHAF_LAYER_HPP


#include <unordered_map>
#include "activations.hpp"
#include <arrayfire.h>

using af::array;

namespace Layers {

    class Layer {
    public:
        std::vector<std::string> description;
        std::vector<std::string> gradients;

        virtual array
        forward(const array &input, std::unordered_map<std::string, array> &cache) { return input; };

        virtual array
        backward(const array &input, std::unordered_map<std::string, array> &cache) { return input; };

        virtual void update_parameters(std::unordered_map<std::string, array> &cache) {};

        virtual std::unordered_map<std::string, std::vector<int>> layer_shapes() {};
    };

    class Dense : public Layer {
    private:
        array W;
        array b;
        Activations::Activation *activation;

    public:

        Dense(int from_size, int to_size, Activations::Activation *activation,
              const std::string &parameter_initialization);

        array forward(const array &input, std::unordered_map<std::string, array> &cache) override;

        array backward(const array &dA, std::unordered_map<std::string, array> &cache) override;

        void update_parameters(std::unordered_map<std::string, array> &cache) override;

        std::unordered_map<std::string, std::vector<int>> layer_shapes() override;
    };




}

#endif //TENSORTORCHAF_LAYER_HPP
