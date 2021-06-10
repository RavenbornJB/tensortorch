//
// Created by bohdansydor on 13.05.21.
//

#include "optimizers.hpp"
#include "model.hpp"
#include <thread>
#include <mutex>

Optimizers::Parallel::Parallel(int _batch_size, double _learning_rate) {
    this->learning_rate = _learning_rate;
    this->batch_size = _batch_size;
}

void optimize_task(Model *model, const MatrixXd &X_train, const MatrixXd &Y_train,
                   std::vector<std::vector<std::unordered_map<std::string, MatrixXd> > > &threads_caches,
                   std::mutex &m) {
    auto thread_cache = std::vector<std::unordered_map<std::string, MatrixXd>>(model->get_layers().size());
    MatrixXd Y_pred = model->forward(X_train, thread_cache);
    model->backward(Y_pred, Y_train, thread_cache);
    {
        std::lock_guard<std::mutex> lg{m};
        threads_caches.push_back(thread_cache);
    }
}


void Optimizers::Parallel::optimize(Model *model, const MatrixXd &X_train, const MatrixXd &Y_train, int num_epochs) {

    std::vector<Layers::Layer *> layers = model->get_layers();

    auto gradients = *(new std::vector<std::unordered_map<std::string, MatrixXd>>(layers.size()));
    for (int l = 0; l < layers.size(); ++l) {
        std::unordered_map<std::string, std::vector<int>> layer_shapes = layers[l]->layer_shapes();
        for (const auto &grad: layers[l]->gradients) {
            gradients[l][grad] = MatrixXd::Zero(layer_shapes[grad][0], layer_shapes[grad][1]);
        }
    }

    int THREADS_NUM = 8;
    std::mutex m;
    std::vector<std::thread> Threads;
    std::vector<std::vector<std::unordered_map<std::string, MatrixXd> > > threads_caches;
//    thread_pool Pool(THREADS_NUM);


    for (int i = 0; i < num_epochs; i++) {

        if (i % 100 == 0 || i == num_epochs - 1) {
            MatrixXd Y_pred = model->predict(X_train);
            double cost = model->compute_cost(Y_pred, Y_train);
            std::cout << "Cost at iteration " << i << ": " << cost << std::endl;
        }

        for (int j = 0; j < (X_train.cols() / (double) (THREADS_NUM * batch_size)); j++) {
            int k = 0;
            while ((k + THREADS_NUM) * batch_size < X_train.cols()) {
//                  Pool.submit(optimize_task, std::ref(model),
                Threads.emplace_back(optimize_task, std::ref(model),
                                     X_train.middleCols(j + k * batch_size, batch_size),
                                     Y_train.middleCols(j + k * batch_size, batch_size),
                                     std::ref(threads_caches), std::ref(m));
                k++;
            }

            for (auto &t: Threads) { t.join(); }
//            Pool.wait_for_tasks();

            for (int l = 0; l < layers.size(); l++) {
                for (const auto &grad: layers[l]->gradients) {
                    MatrixXd sum_matrix = MatrixXd::Zero(layers[l]->layer_shapes()[grad][0],
                                                         layers[l]->layer_shapes()[grad][1]);
                    for (int c = 0; c < threads_caches.size(); c++) {
                        sum_matrix += threads_caches[c][l][grad];
                    }
                    gradients[l][grad] = sum_matrix / threads_caches.size();
                }
            }

            update_parameters(model->get_layers(), gradients);
            Threads.clear();
            threads_caches.clear();

        }
    }
}


void Optimizers::Parallel::update_parameters(std::vector<Layers::Layer *> &layers,
                                             std::vector<std::unordered_map<std::string, MatrixXd>> &cache) {
    for (int l = 0; l < layers.size(); ++l) {
        for (const auto &grad: layers[l]->gradients) {
            cache[l][grad] *= learning_rate;
        }
        layers[l]->update_parameters(cache[l]);
    }
}
