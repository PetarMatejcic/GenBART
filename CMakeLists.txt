cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_backend src/cpp/packed_forest.cpp)
target_compile_features(_backend PRIVATE cxx_std_17)

install(TARGETS _backend LIBRARY DESTINATION genbart)
