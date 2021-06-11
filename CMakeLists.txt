cmake_minimum_required(VERSION 3.19)
project(tensortorchAf)

set(CMAKE_CXX_STANDARD 20)

include_directories(inc)


find_package(ArrayFire REQUIRED)

INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})

add_executable(af_test src/main.cpp inc/activations.hpp inc/layer.hpp src/dense.cpp inc/losses.hpp inc/model.hpp src/model.cpp inc/optimizers.hpp src/BGD.cpp)

# To use Unified backend, do the following.
# Unified backend lets you choose the backend at runtime
TARGET_LINK_LIBRARIES(af_test ${ArrayFire_LIBRARIES} )
