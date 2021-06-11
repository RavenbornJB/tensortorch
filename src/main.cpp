#include <iostream>
#include <fstream>

#include "tensortorch.hpp"


std::unordered_map<int, std::string> read_vocabulary(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("File cannot be opened");
    }

    std::unordered_map<int, std::string> vocabulary;
    std::string token;
    std::string id;
    std::string word;
    while (getline(file, token)) {
        std::istringstream iss(token);
        iss >> id;
        iss >> word;
        vocabulary[std::stoi(id)] = word;
    }

    return vocabulary;
}

std::pair<std::vector<std::vector<int>>, std::vector<int>> get_imdb_data(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("File cannot be opened");
    }

    std::vector<std::vector<int>> reviews;
    std::vector<int> labels;

    std::string token;
    getline(file, token);
    int num_reviews = std::stoi(token);

    for (int i = 0; i < num_reviews; ++i) {
        std::vector<int> review;
        int label;

        getline(file, token);
        std::istringstream iss(token);
        std::string review_word;
        while(iss >> review_word) {
            review.push_back(std::stoi(review_word));
        }
        getline(file, token);
        label = std::stoi(token);

        reviews.emplace_back(review);
        labels.emplace_back(label);
    }

    return std::make_pair(reviews, labels);
}

MatrixXd int_vector_to_one_hot_matrix(const std::vector<int>& v, int len, int vocab_size) {
    MatrixXd m = MatrixXd::Zero(vocab_size, len);

    for (int c = 0; c < v.size(); ++c) {
        m(v[c], c) = 1;
    }

    return m;
}

std::vector<MatrixXd> create_imdb_matrix(const std::vector<std::vector<int>>& reviews, int vocab_size, int minibatch_size) {
    std::vector<MatrixXd> res;

    for (int b = 0; b < reviews.size() / minibatch_size; ++b) {
        int max_len = 0;
        for (int r = 0; r < minibatch_size; ++r) {
            max_len = std::max(max_len, (int) reviews[b * minibatch_size + r].size());
        }
        MatrixXd minibatch(vocab_size * minibatch_size, max_len);

        for (int r = 0; r < minibatch_size; ++r) {
            auto sentence = int_vector_to_one_hot_matrix(reviews[b * minibatch_size + r], max_len, vocab_size);

            for (int i = 0; i < vocab_size; ++i) {
                for (int j = 0; j < max_len; ++j) {
                    minibatch(r * vocab_size + i, j) = sentence(i, j);
                }
            }
        }

        res.push_back(minibatch);
    }

    return res;
}

std::vector<MatrixXd> create_imdb_label_matrix(const std::vector<int>& labels, int minibatch_size) {
    std::vector<MatrixXd> res;

    for (int b = 0; b < labels.size() / minibatch_size; ++b) {
        MatrixXd label_matrix(minibatch_size, 1);

        for (int r = 0; r < minibatch_size; ++r) {
            label_matrix(r, 0) = labels[b * minibatch_size + r];
        }

        res.push_back(label_matrix);
    }

    return res;
}

int prediction_to_index(const Eigen::VectorXd& input, double threshold) {
    for (int r = 0; r < input.size(); ++r) {
        if (input(r) > threshold) return r;
    }
    return -1;
}

Eigen::VectorXd prediction_matrix_to_vector(const MatrixXd& input, double threshold) {
    Eigen::VectorXd res(input.cols());
    for (int c = 0; c < input.cols(); ++c) {
        res(c) = prediction_to_index(input.col(c), threshold);
    }
    return res;
}

void matrix_to_file(const MatrixXd& matrix, const std::string& filename) {
    std::ofstream file(filename);

    file << std::to_string(matrix.rows()) << " " << std::to_string(matrix.cols()) << "\n";
    file << matrix;
}

MatrixXd matrix_from_file(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("File cannot be opened");
    }

    int rows;
    int cols;
    file >> rows >> cols;

    MatrixXd m(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> m(i, j);
        }
    }

    return m;
}

MatrixXd get_sentence(const MatrixXd& sentences, int sentence_num) {
    return sentences.block(600 * sentence_num, 0, 600, 30);
}

int main() {
//    auto vocabulary = read_vocabulary("data_generation/vocabulary.txt");
//
//    auto train_data = get_imdb_data("data_generation/data_imdb_train.txt");
//    auto X_train = train_data.first;
//    auto Y_train = train_data.second;
//    auto test_data = get_imdb_data("data_generation/data_imdb_test.txt");
//    auto X_test = test_data.first;
//    auto Y_test = test_data.second;
//
//    int vocab_size = 800;
//
//    int minibatches = 200;
//    int minibatch_size = 10;
//    std::cout << "Creating X_train..." << std::endl;
//    std::vector<MatrixXd> X_train_minibatches = create_imdb_matrix(X_train, vocab_size, minibatch_size);
//    std::cout << "Creating Y_train..." << std::endl;
//    std::vector<MatrixXd> Y_train_minibatches = create_imdb_label_matrix(Y_train, minibatch_size);
//
//    // --- HYPERPARAMETERS ---
//
//    int hidden_size = 50;
//    int num_epochs = 5;
//    double reg_param = -1;
//    double learning_rate = 0.001;
////    double momentum = 0.9;
//    double beta1 = 0.9;
//    double beta2 = 0.999;
//
//    // --- MODEL CREATION ---
//
//    std::vector<Layers::Layer*> layers = {
//            new Layers::RNN(vocab_size, hidden_size, 1,
//                        new Activations::Tanh, new Activations::Sigmoid,
//                        "he", false)
//    };
//
//    Model rnn_model(layers, reg_param);
//
//    rnn_model.compile(
//            new Losses::BinaryCrossentropy,
//            new Optimizers::Adam(minibatch_size, learning_rate, beta1, beta2)
//            );
//
//    for (int epoch = 0; epoch < num_epochs; ++ epoch) {
//        std::cout << "Epoch " << epoch << ":" << std::endl;
//        for (int i = 0; i < minibatches; ++i) {
//            rnn_model.fit(X_train_minibatches[i], Y_train_minibatches[i], 1);
//        }
//    }
//
//    rnn_model.save("models/sentiment-analysis");

//    int vocab_size = 600;
//    int words = 30;
//    int sentences = 150;
//
//    MatrixXd X_train = MatrixXd::Zero(sentences * vocab_size, words);
//    MatrixXd Y_train = MatrixXd::Zero(sentences, 1);
//
//    std::random_device rand_dev;
//    std::mt19937 gen(rand_dev());
//
//    std::uniform_int_distribution<int> dist(0, vocab_size - 1);
//    std::uniform_int_distribution<int> dist_words(words / 2 + 1, words);
//    for (int s = 0; s < sentences; ++s) {
//        for (int w = 0; w < dist_words(gen); ++w) {
//            X_train(s * vocab_size + dist(gen), w) = 1;
//        }
//    }
//
//    std::uniform_int_distribution<int> dist_y(0, 1);
//    for (int s = 0; s < sentences; ++s) {
//        Y_train(s, 0) = dist_y(gen);
//    }
//
//    std::vector<Layers::Layer*> layers = {
//        new Layers::RNN(vocab_size, 50, 1,
//            new Activations::Tanh, new Activations::Sigmoid,
//            "he", false)
//    };
//
//    Model rnn_model(layers, 0.01);
//
//    rnn_model.compile(
//            new Losses::BinaryCrossentropy,
//            new Optimizers::Adam(1, 0.01, 0.9, 0.999)
//        );
//
//    rnn_model.fit(X_train, Y_train, 200);
//
//    rnn_model.save("randomized-sentiment-analysis");
//
//    matrix_to_file(X_train, "models/randomized-sentiment-analysis-X_train");
//    matrix_to_file(Y_train, "models/randomized-sentiment-analysis-Y_train");
    MatrixXd X = matrix_from_file("models/randomized-sentiment-analysis-X_train");
    MatrixXd Y = matrix_from_file("models/randomized-sentiment-analysis-Y_train");

    Model model = Model::Load("randomized-sentiment-analysis");

    int sentences = 150;

    int correct = 0;
    for (int s = 0; s < sentences; ++s) {
        MatrixXd res = model.predict(get_sentence(X, s));

        int prediction = (int) std::round(res(0, 0));
        int ground_truth = (int) Y(s, 0);

        if (prediction == ground_truth) ++correct;
    }

    std::cout << "Accuracy: " << correct * 100. / sentences << "%" << std::endl;
}
