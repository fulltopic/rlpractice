/*
 * cartacnet.cpp
 *
 *  Created on: Apr 30, 2021
 *      Author: zf
 */


#include "gymtest/lunarnets/netconfig.h"
#include <torch/torch.h>
#include "gymtest/lunarnets/cartacnet.h"

#include <string.h>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>


CartACFcNet::CartACFcNet(int input, int output):
	inputNum(input),
	outputNum(output),
	fc0(input, 256),
	vOutput(256, 1),
	aOutput(256, output){
	register_module("fc0", fc0);
	register_module("vOutput", vOutput);
	register_module("aOutput", aOutput);

//	init_weights(fc0->named_parameters(), sqrt(2.0), 0);
//	init_weights(fcOutput->named_parameters(), sqrt(2.0), 0);
}

std::vector<torch::Tensor> CartACFcNet::forward(torch::Tensor input) {
	torch::Tensor output = fc0->forward(input);
	output = torch::leaky_relu(output);


	auto v = vOutput->forward(output);
	auto a = aOutput->forward(output);

	return {a, v};
}


