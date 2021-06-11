//
// Created by bohdan on 11.06.21.
//

#ifndef TENSORTORCHAF_MODEL_HPP
#define TENSORTORCHAF_MODEL_HPP

#include "layer.hpp"
#include "losses.hpp"
#include "optimizers.hpp"

class Model {
private:
    int L;
    std::vector<Layers::Layer*> layers;

    Losses::Loss* loss;
    Optimizers::Optimizer* optimizer;

public:
    explicit Model(std::vector<Layers::Layer*> &layers);

    void compile(Losses::Loss* loss, Optimizers::Optimizer* optimizer);

    void fit(const array& X_train, const array& Y_train, int num_epochs);

    array predict(const array& X_test);

    std::vector<Layers::Layer*>& get_layers();

    array forward(const array &input, std::vector<std::unordered_map<std::string, array>> &thread_cache);

    array compute_cost(const array &y_pred, const array &y_true);

    void backward(const array &y_pred, const array &y_true, std::vector<std::unordered_map<std::string, array>> &thread_cache);

};


#endif //TENSORTORCHAF_MODEL_HPP
