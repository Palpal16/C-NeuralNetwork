//
// Created by Danilo Ardagna on 01/11/23.
//

#ifndef NEURALNETWORKS_NEURON_H
#define NEURALNETWORKS_NEURON_H
#include "dense_matrix.hpp"
#include "utils.h"

class neuron {
private:
    la::dense_matrix weights;
    double bias;
    ptr_act_function p_act_func;

public:
    neuron(size_t input_size, const ptr_act_function & p_a_f);

    double eval (const la::dense_matrix & input_vect) const;

};


#endif //NEURALNETWORKS_NEURON_H
