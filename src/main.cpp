#include <iostream>

#include "model.h"

int main() {
    std::vector<size_t> ldims = {5, 7, 4, 1};
    Model model(ldims);
    model.print_layers();
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
