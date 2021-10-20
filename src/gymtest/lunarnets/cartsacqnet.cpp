/*
 * cartsacqnet.cpp
 *
 *  Created on: Sep 16, 2021
 *      Author: zf
 */


#include "gymtest/lunarnets/netconfig.h"
#include <torch/torch.h>
#include "gymtest/lunarnets/cartsacqnet.h"
#include "gymtest/utils/netinitutils.h"

#include <string.h>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>


CartSacQNet::CartSacQNet(int input, int output):
	inputNum(input),
	outputNum(output),
	fc0(input, 256),
	fcOutput(256, output) {
	register_module("fc0", fc0);
	register_module("fcOutput", fcOutput);

	NetInitUtils::Init_weights(fc0->named_parameters(), sqrt(2.0), 0, NetInitUtils::Kaiming);
}

torch::Tensor CartSacQNet::forward(torch::Tensor input) {
	torch::Tensor output = fc0->forward(input);
	output = torch::relu(output);
	output = fcOutput->forward(output);
//	std::cout << "output before relu: " << output << std::endl;
//	output = torch::relu(output);

	return output;
}


