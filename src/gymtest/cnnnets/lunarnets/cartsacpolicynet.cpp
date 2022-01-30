/*
 * cartsacpolicynet.cpp
 *
 *  Created on: Sep 16, 2021
 *      Author: zf
 */




#include "gymtest/cnnnets/lunarnets/netconfig.h"
#include <torch/torch.h>
#include "gymtest/cnnnets/lunarnets/cartsacpolicynet.h"

#include <string.h>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>


CartSacPolicyNet::CartSacPolicyNet(int input, int output):
	inputNum(input),
	outputNum(output),
	fc0(input, 256),
	fcOutput(256, output) {
	register_module("fc0", fc0);
	register_module("fcOutput", fcOutput);

}

torch::Tensor CartSacPolicyNet::forward(torch::Tensor input) {
	torch::Tensor output = fc0->forward(input);
	output = torch::relu(output);
	output = fcOutput->forward(output);
//	output = torch::relu(output);

	return output;
}
