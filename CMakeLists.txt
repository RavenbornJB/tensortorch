cmake_minimum_required(VERSION 3.17)
project(neuralnet_lib)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

if (${CMAKE_CXX_COMPILER_ID} MATCHES Clang)
    add_compile_options(-O2)
elseif (${CMAKE_CXX_COMPILER_ID} MATCHES GNU)
    add_compile_options(-O2)
elseif (${CMAKE_CXX_COMPILER_ID} MATCHES MSVC)
    add_compile_options(/O2)
endif()

#option(WARNINGS_AS_ERRORS   "Treat compiler warnings as errors."    ON)
#option(ENABLE_CONAN         "Use Conan as a package manager."       OFF)
#option(ENABLE_PVS_STUDIO    "Use PVS-Studio static code analyzer."  OFF) # Option for the local usage only. PVS-Studio isn't installed on GitHub machines.
#option(ENABLE_SANITIZERS    "Use sanitizers to detect errors."      OFF) # Option for the test builds. Do not use it in the production builds.

include_directories(inc src)
add_executable(neuralnet_lib src/main.cpp src/linalg.cpp src/layer.cpp src/model.cpp src/utility.cpp)
