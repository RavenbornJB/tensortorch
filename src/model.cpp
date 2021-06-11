//
// Created by raven on 5/4/21.
//

#include "model.hpp"
#include "optimizers.hpp"


Model::Model(std::vector<Layers::Layer*> &layers, double regularization_parameter) {
    this->L = (int) layers.size();
    this->layers = layers;
    this->regularization_parameter = regularization_parameter;

    this->compiled = false;
}


Activations::Activation* Model::make_activation(const std::string& activation_name) {
    if (activation_name == "linear") {
        return new Activations::Linear;
    } else if (activation_name == "sigmoid") {
        return new Activations::Sigmoid;
    } else if (activation_name == "softmax") {
        return new Activations::Softmax;
    } else if (activation_name == "tanh") {
        return new Activations::Tanh;
    } else if (activation_name == "relu") {
        return new Activations::Relu;
    } else {
        throw std::logic_error("Invalid loss name");
    }
}

Losses::Loss* Model::make_loss(const std::string& loss_name) { // TODO create literally anything better here, like a map
    if (loss_name == "binary_crossentropy") {
        return new Losses::BinaryCrossentropy;
    } else if (loss_name == "categorical_crossentropy") {
        return new Losses::CategoricalCrossentropy;
    } else if (loss_name == "mean_squared_error") {
        return new Losses::MSE;
    } else {
        throw std::logic_error("Invalid loss name");
    }
}

Optimizers::Optimizer* Model::make_optimizer(const std::string& optimizer_name, const std::vector<double>& params) {
    if (optimizer_name == "bgd") {
        return new Optimizers::BGD(params[0]);
    } else if (optimizer_name == "sgd") {
        return new Optimizers::SGD((int) params[0], params[1], params[2]);
    } else if (optimizer_name == "rmsprop") {
        return new Optimizers::RMSprop((int) params[0], params[1], params[2]);
    } else if (optimizer_name == "adam") {
        return new Optimizers::Adam((int) params[0], params[1], params[2], params[3]);
    } else if (optimizer_name == "parallel") {
        return new Optimizers::Parallel((int) params[0], params[1]);
    } else {
        throw std::logic_error("Invalid optimizer name");
    }
}

void Model::save(const std::string& filename) {
    if (!compiled) {
        throw std::logic_error("Model can not be saved before compilation");
    }

    // clear file and write model metadata
    std::ofstream file(filename + ".ttwf", std::ios::out | std::ios::trunc);
    file << L << " " << loss->get_name() << " " << optimizer->get_name() << " " << regularization_parameter << "\n";
    for (double param: optimizer->get_params()) {
        file << param << " ";
    }
    file << "\n";
    file.close();

    for (const auto& layer: layers) {
        layer->save(filename + ".ttwf");
    }
}

Model Model::Load(const std::string& filename) {
    std::ifstream file(filename + ".ttwf");

    if (!file.is_open()) {
        throw std::runtime_error("Load file cannot be opened or does not exist");
    }

    try {
        std::string info;

        // first line, general model information
        std::vector<std::string> model_metadata(4);
        for (int i = 0; i < 4; ++i) {
            file >> model_metadata[i];
        }
        file.ignore();

        // second line, optimizer parameters, variable amount of them so getline
        std::string token;
        getline(file, info);
        std::istringstream iss_opt(info);
        std::vector<double> opt_params;
        while (iss_opt >> token) {
            opt_params.push_back(std::stod(token));
        }

        // create layer vector, loss, optimizer, set reg. param.
        std::vector<Layers::Layer*> layers(std::stoi(model_metadata[0]));
        Losses::Loss* loss = make_loss(model_metadata[1]);
        Optimizers::Optimizer* optimizer = make_optimizer(model_metadata[2], opt_params);
        double regularization_parameter = std::stod(model_metadata[3]);

        // populate layer vector
        for (int l = 0; l < std::stoi(model_metadata[0]); ++l) {
            getline(file, info);
            std::istringstream iss_layer(info);
            std::vector<std::string> layer_metadata;
            while (iss_layer >> token) {
                layer_metadata.push_back(token);
            }

            // temp layer switch TODO refactor
            if (layer_metadata[0] == "dense") {
                int input_size = std::stoi(layer_metadata[1]);
                int output_size = std::stoi(layer_metadata[2]);
                Activations::Activation* activation = make_activation(layer_metadata[3]);
                std::string parameter_initialization = layer_metadata[4];

                layers[l] = new Layers::Dense(input_size, output_size, activation, parameter_initialization);

                MatrixXd W(output_size, input_size);
                MatrixXd b(output_size, 1);
                double num;
                for (int i = 0; i < output_size; ++i) {
                    for (int j = 0; j < input_size; ++ j) {
                        file >> num;
                        W(i, j) = num;
                    }
                }
                for (int i = 0; i < output_size; ++i) {
                    for (int j = 0; j < 1; ++ j) {
                        file >> num;
                        b(i, j) = num;
                    }
                }
                layers[l]->set_parameters({W, b});
                // ignore \n for next layer
                file.ignore();

            } else if (layer_metadata[0] == "rnn") {
                int input_size = std::stoi(layer_metadata[1]);
                int hidden_size = std::stoi(layer_metadata[2]);
                int output_size = std::stoi(layer_metadata[3]);
                Activations::Activation* activation_a = make_activation(layer_metadata[4]);
                Activations::Activation* activation_y = make_activation(layer_metadata[5]);
                std::string parameter_initialization = layer_metadata[6];
                bool return_sequences = (layer_metadata[7] == "1");

                layers[l] = new Layers::RNN(input_size, hidden_size, output_size,
                                            activation_a, activation_y,
                                            parameter_initialization, return_sequences);

                MatrixXd Waa(hidden_size, hidden_size);
                MatrixXd Wax(hidden_size, input_size);
                MatrixXd Wya(output_size, hidden_size);
                MatrixXd ba(hidden_size, 1);
                MatrixXd by(output_size, 1);
                double num;
                for (int i = 0; i < hidden_size; ++i) {
                    for (int j = 0; j < hidden_size; ++ j) {
                        file >> num;
                        Waa(i, j) = num;
                    }
                }
                for (int i = 0; i < hidden_size; ++i) {
                    for (int j = 0; j < input_size; ++ j) {
                        file >> num;
                        Wax(i, j) = num;
                    }
                }
                for (int i = 0; i < output_size; ++i) {
                    for (int j = 0; j < hidden_size; ++ j) {
                        file >> num;
                        Wya(i, j) = num;
                    }
                }
                for (int i = 0; i < hidden_size; ++i) {
                    for (int j = 0; j < 1; ++ j) {
                        file >> num;
                        ba(i, j) = num;
                    }
                }
                for (int i = 0; i < output_size; ++i) {
                    for (int j = 0; j < 1; ++ j) {
                        file >> num;
                        by(i, j) = num;
                    }
                }
                layers[l]->set_parameters({Waa, Wax, Wya, ba, by});

            } else throw std::logic_error("Unknown layer type");
        }

        // create model
        Model model(layers, regularization_parameter);
        model.compile(loss, optimizer);
        return model;

    } catch (...) {
        throw std::runtime_error("Load file is corrupted");
    }
}

MatrixXd Model::forward(const MatrixXd &input, std::vector<std::unordered_map<std::string, MatrixXd>> &thread_cache) {
    if (!compiled) {
        throw std::logic_error("Model is not compiled yet");
    }

    MatrixXd y_pred(input);

    for (int l = 0; l < L; ++l) {
        y_pred = layers[l]->forward(y_pred, thread_cache[l]);
    }

    return y_pred;
}


double Model::compute_cost(const MatrixXd &y_pred, const MatrixXd &y_true) {
    MatrixXd losses = loss->loss(y_pred, y_true);
    return losses.mean();
}


void Model::backward(const MatrixXd &y_pred, const MatrixXd &y_true,
                     std::vector<std::unordered_map<std::string, MatrixXd>> &thread_cache) {
    MatrixXd dA = this->loss->loss_back(y_pred, y_true);
    for (int l = L - 1; l >= 0; --l) {
        dA = layers[l]->backward(dA, thread_cache[l], regularization_parameter);
    }
}


void Model::fit(const MatrixXd& X_train, const MatrixXd& Y_train, int num_epochs) {
    //TODO add shapes check
    optimizer->optimize(this, X_train, Y_train, num_epochs);
}


MatrixXd Model::predict(const MatrixXd& X_test) {
    auto thread_cache = std::vector<std::unordered_map<std::string, MatrixXd>>(L);
    return forward(X_test, thread_cache);
}


std::vector<Layers::Layer *>& Model::get_layers() {
    return this->layers;
}


void Model::compile(Losses::Loss* _loss, Optimizers::Optimizer* _optimizer) {
    if (_optimizer->get_name() == "parallel" && layers[0]->description[0] == "rnn") {
        throw std::logic_error("Parallel optimizer currently not supported with RNN layers");
    }

    loss = _loss;
    optimizer = _optimizer;
    compiled = true;
}

std::string Model::summary() {
    return std::string();
}
