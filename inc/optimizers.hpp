//
// Created by bohdan on 11.06.21.
//

#ifndef TENSORTORCHAF_OPTIMIZERS_HPP
#define TENSORTORCHAF_OPTIMIZERS_HPP

#include "layer.hpp"

class Model;

namespace Optimizers {
    class Optimizer {
    private:
        int batch_size;
    public:
        virtual void optimize(Model *model, const array &X_train, const array &Y_train, int num_epochs);

        virtual void update_parameters(
                std::vector<Layers::Layer *> &layers,
                std::vector <std::unordered_map<std::string, array>> &cache,
                std::vector <std::unordered_map<std::string, array>> &rms_cache);
    };

    class BGD : public Optimizer {
    private:
        double learning_rate;
    public:
        explicit BGD(double _learning_rate);

        void optimize(Model *model, const array &X_train, const array &Y_train, int num_epochs) override;

        void update_parameters(std::vector<Layers::Layer *> &layers,
                               std::vector <std::unordered_map<std::string, array>> &cache) ;
    };
}

#endif //TENSORTORCHAF_OPTIMIZERS_HPP
