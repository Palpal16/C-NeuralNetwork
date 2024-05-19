//
// Created by Danilo Ardagna on 01/11/23.
//

#include "sigmoid.h"

double sigmoid::eval(double x) {
    return 1/(1+ std::exp(-x));
}
