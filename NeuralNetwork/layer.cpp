//
// Created by Danilo Ardagna on 01/11/23.
//

#include "layer.h"

layer::layer(size_t input_size, size_t output_size, const ptr_act_function &p_a_f) {
    this->input_size=input_size;
    this->output_size=output_size;

    for (size_t i = 0; i < output_size; ++i)
        neurons.emplace_back(input_size,p_a_f);
}

la::dense_matrix layer::eval(const la::dense_matrix & input_vector) const{
    la::dense_matrix output_vec(output_size,1);

    for (size_t i = 0; i < output_size; ++i) {
        output_vec(i, 0) = neurons[i].eval(input_vector);
    }
    return output_vec;
}

size_t layer::get_input_size() const {
    return input_size;
}

size_t layer::get_output_size() const {
    return output_size;
}
