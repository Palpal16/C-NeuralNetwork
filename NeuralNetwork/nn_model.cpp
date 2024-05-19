//
// Created by Danilo Ardagna on 01/11/23.
//

#include "nn_model.h"

la::dense_matrix nn_model::predict(const la::dense_matrix &input_vector) const {

    la::dense_matrix vec(input_vector.rows(), input_vector.columns(), *input_vector.data());

    for( const layer & l : layers){
        vec = l.eval(vec);
    }
    return vec;
}

void nn_model::add_layer(const layer & l) {

    layers.push_back(l);

}
