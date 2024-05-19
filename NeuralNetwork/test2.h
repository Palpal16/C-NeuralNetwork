//
// Created by Danilo Ardagna on 01/11/23.
//

#ifndef NEURALNETWORKS_TEST2_H
#define NEURALNETWORKS_TEST2_H
#include <memory>
#include <iostream>
#include "dense_matrix.hpp"
#include "utils.h"
#include "neuron.h"
#include "activation_function.h"
#include "sigmoid.h"

la::dense_matrix input_vec (4,1,1);
ptr_act_function paf = std::make_shared<sigmoid>();

neuron n(4,paf);

std::cout << n.eval(input_vec) << std::endl;
#endif //NEURALNETWORKS_TEST2_H
