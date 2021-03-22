cmake_minimum_required(VERSION 3.17)
project(nn_project)

set(CMAKE_CXX_STANDARD 14)

add_executable(nn_project src/main.cpp src/model.cpp src/model.h src/linalg.cpp src/linalg.h src/layer.cpp src/layer.h)