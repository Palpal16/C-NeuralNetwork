//
// Created by Danilo Ardagna on 01/11/23.
//

#ifndef NEURALNETWORKS_TEST1_H
#define NEURALNETWORKS_TEST1_H
#include <iostream>
#include <memory>
#include "activation_function.h"
#include "sigmoid.h"
#include "tanh_af.h"
#include "re_lu.h"

using std::shared_ptr;
using std::make_shared;
shared_ptr<activation_function> sa;

sa =make_shared<tanh_af>();

std::cout << round(sa->eval(-200)) << std::endl;

sa =make_shared<sigmoid>();

std::cout << round(sa->eval(-200))<< std::endl;

sa =make_shared<re_lu>();

std::cout << round(sa->eval(-200)) << std::endl;
#endif //NEURALNETWORKS_TEST1_H
