cmake_minimum_required(VERSION 3.17)
project(nn_project)

set(CMAKE_CXX_STANDARD 20)

add_executable(nn_project src/main.cpp src/linalg.cpp src/linalg.h)