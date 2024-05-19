//
// Created by Danilo Ardagna on 01/11/23.
//

#ifndef NEURALNETWORKS_TEST3_H
#define NEURALNETWORKS_TEST3_H

#include <iostream>
#include "dense_matrix.hpp"
#include "utils.h"
#include "layer.h"
#include "activation_function.h"
#include "sigmoid.h"

la::dense_matrix input_vec (4,1,1);

ptr_act_function paf = std::make_shared<sigmoid>();

layer l(4,3,paf);
la::dense_matrix output_vec(3,1);
output_vec = l.eval(input_vec);

output_vec.print(std::cout);

#endif //NEURALNETWORKS_TEST3_H
