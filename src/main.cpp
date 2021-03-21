#include <iostream>

#include "model.h"

int main() {
    std::vector<size_t> ldims = {1, 2, 3};
    Model model(ldims);
    model.print_layers();
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
