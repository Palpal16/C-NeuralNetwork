//
// Created by Danilo Ardagna on 01/11/23.
//

#ifndef NEURALNETWORKS_TEST4_H
#define NEURALNETWORKS_TEST4_H
#include "nn_model.h"
#include "re_lu.h"
#include "dense_matrix.hpp"

nn_model model;

auto paf=std::make_shared<re_lu>();
la::dense_matrix input_vec (4,1,400);

model.add_layer(layer(4,4,paf));

model.add_layer(layer(4,3,paf));

model.add_layer(layer(3,2,paf));

la::dense_matrix output_vec = model.predict(input_vec);

output_vec.print(std::cout);
#endif //NEURALNETWORKS_TEST4_H
