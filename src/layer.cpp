//
// Created by raven on 3/22/21.
//

#include "layer.h"

/* Тут визначається тип функції активації, а тоді ініціалізуються параметри W та b.
 * W ініціалізується випадковими числами з нормального розподілу (mean = 0, var = 2 / from_size) за методом
 * He initialization.
 * b ініціалізується нулями.
 * */
Layer::Layer(const std::string& activation_type, size_t from_size, size_t to_size) {
    if (activation_type == "sigmoid") {
        this->activation = Layer::sigmoid;
    } else if (activation_type == "tanh") {
        this->activation = Layer::tanh;
    } else if (activation_type == "relu") {
        this->activation = Layer::relu;
    } else {
        throw std::logic_error("Activation type not allowed");
    }

    std::default_random_engine gen{static_cast<long unsigned int>(time(nullptr))};
    std::normal_distribution<double> dist(0., std::sqrt(2. / from_size)); // He initialization
    for (size_t i = 0; i < to_size; ++i) {
        std::vector<double> row;
        row.reserve(from_size);
        for (size_t j = 0; j < from_size; ++j) {
            row.push_back(dist(gen));
        }
        W.push_back(row);
        b.push_back(0);
    }
}

/* Дебаг-функція для друку параметрів
 */
void Layer::print_parameters() {
    std::cout << "W: " << std::endl;
    for (auto & row : W) {
        for (double el : row) std::cout << el << " ";
        std::cout << std::endl;
    }

    std::cout << "\nb: " << std::endl;
    for (double el: b) std::cout << el << "\n";
    std::cout << std::endl;
}

/* Плейсхолдери для функцій активації
 * TODO визначитись, чи будуть вони тут, чи в linalg
 */
std::vector<std::vector<double>> Layer::sigmoid(const std::vector<std::vector<double>> &input) {
    std::vector<std::vector<double>> output;
    for (int i = 0; i < input.size(); ++i) {
        std::vector<double> row;
        row.reserve(input[0].size());
        for (int j = 0; j < input[0].size(); ++j) {
            row.push_back(1 / (1 + std::exp(- input[i][j])));
        }
        output.push_back(row);
    }
    return output;
}
std::vector<std::vector<double>> Layer::tanh(const std::vector<std::vector<double>> &input) {
    std::vector<std::vector<double>> output;
    for (int i = 0; i < input.size(); ++i) {
        std::vector<double> row;
        row.reserve(input[0].size());
        for (int j = 0; j < input[0].size(); ++j) {
            row.push_back(std::tanh(input[i][j]));
        }
        output.push_back(row);
    }
    return output;
}
std::vector<std::vector<double>> Layer::relu(const std::vector<std::vector<double>> &input) {
    std::vector<std::vector<double>> output;
    for (int i = 0; i < input.size(); ++i) {
        std::vector<double> row;
        row.reserve(input[0].size());
        for (int j = 0; j < input[0].size(); ++j) {
            row.push_back(input[i][j] * (int)(input[i][j] > 0));
        }
        output.push_back(row);
    }
    return output;
}

/* Функція лінійного кроку: Z = W * X + b
 * TODO реалізувати через функції з linalg, коли вони будуть готові
 */
std::vector<std::vector<double>> Layer::linear(const std::vector<std::vector<double>> &input) {
    return input;
}

/* Функція активації
 * викликає один з варіантів активації:
 * sigmoid, tanh, relu (залежно від параметра в конструкторі)
 */

std::vector<std::vector<double>> Layer::forward(const std::vector<std::vector<double>> &input) {
    return this->activation(this->linear(input));
}

//int main() {
//    Layer test_layer("relu", 5, 3);
//    auto res = test_layer.forward({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {-1, -2, -3}, {-4, -5, -6}});
//    for (const auto& el: res) {
//        for (auto e: el) {
//            std::cout << e << " ";
//        }
//        std::cout << std::endl;
//    }
//}
