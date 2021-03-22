cmake_minimum_required(VERSION 3.17)
project(nn_project)

set(CMAKE_CXX_STANDARD 14)

add_executable(nn_project src/main.cpp inc/linalg.cpp src/linalg.h)