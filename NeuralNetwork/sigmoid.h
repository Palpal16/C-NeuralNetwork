//
// Created by Danilo Ardagna on 01/11/23.
//

#ifndef NEURALNETWORKS_SIGMOID_H
#define NEURALNETWORKS_SIGMOID_H
#include "activation_function.h"


class sigmoid : public activation_function{
public:
    double eval(double x) override;
};


#endif //NEURALNETWORKS_SIGMOID_H
