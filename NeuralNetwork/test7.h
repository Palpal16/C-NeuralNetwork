//
// Created by Danilo Ardagna on 01/11/23.
//

#ifndef NEURALNETWORKS_TEST4_H
#define NEURALNETWORKS_TEST4_H
#include "nn_model.h"
#include "re_lu.h"
#include "tanh_af.h"
#include "sigmoid.h"
#include "dense_matrix.hpp"

nn_model model;

auto paf1=std::make_shared<re_lu>();
auto paf2=std::make_shared<tanh_af>();
auto paf3=std::make_shared<sigmoid>();

la::dense_matrix input_vec (4,1,400);

model.add_layer(layer(4,4,paf2));

model.add_layer(layer(4,4,paf1));

model.add_layer(layer(4,2,paf2));

model.add_layer(layer(2,2,paf1));

model.add_layer(layer(2,2,paf2));

model.add_layer(layer(2,1,paf3));

la::dense_matrix output_vec = model.predict(input_vec);

output_vec.print(std::cout);
#endif //NEURALNETWORKS_TEST4_H
