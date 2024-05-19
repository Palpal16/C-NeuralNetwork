//
// Created by Danilo Ardagna on 01/11/23.
//

#ifndef NEURALNETWORKS_LAYER_H
#define NEURALNETWORKS_LAYER_H

#include <memory>
#include "activation_function.h"
#include "neuron.h"
#include "utils.h"
#include "dense_matrix.hpp"



class layer {
private:
    size_t input_size;
    size_t output_size;
    std::vector<neuron>  neurons;
public:

    layer(size_t input_size, size_t output_size, const ptr_act_function & p_a_f);
    la::dense_matrix eval(const la::dense_matrix & input_vector) const;

    size_t get_input_size() const;
    size_t get_output_size() const;
};


#endif //NEURALNETWORKS_LAYER_H
