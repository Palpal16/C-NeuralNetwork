//
// Created by Danilo Ardagna on 01/11/23.
//

#ifndef NEURALNETWORKS_TANH_AF_H
#define NEURALNETWORKS_TANH_AF_H
#include "activation_function.h"


class tanh_af : public activation_function{
public:
    double eval(double x) override;
};



#endif //NEURALNETWORKS_TANH_AF_H
