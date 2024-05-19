cmake_minimum_required(VERSION 3.26)
project(NeuralNetworks)

set(CMAKE_CXX_STANDARD 11)

add_executable(NeuralNetworks main.cpp
        activation_function.cpp
        activation_function.h
        re_lu.cpp
        re_lu.h
        sigmoid.cpp
        sigmoid.h
        tanh_af.cpp
        tanh_af.h
        test1.h
        dense_matrix.hpp
        dense_matrix.cpp
        utils.h
        neuron.cpp
        neuron.h
        layer.cpp
        layer.h
        test2.h
        test3.h
        nn_model.cpp
        nn_model.h
        test4.h
)
