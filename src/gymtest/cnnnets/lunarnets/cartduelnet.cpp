/*
 * cartduelnet.cpp
 *
 *  Created on: Apr 25, 2021
 *      Author: zf
 */


#include "gymtest/cnnnets/lunarnets/netconfig.h"
#include <torch/torch.h>
#include "gymtest/cnnnets/lunarnets/cartduelnet.h"

#include <string.h>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>


CartDuelFcNet::CartDuelFcNet(int input, int output):
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

torch::Tensor CartDuelFcNet::forward(torch::Tensor input) {
	torch::Tensor output = fc0->forward(input);
	output = torch::relu(output);


	auto v = vOutput->forward(output);
	auto a = aOutput->forward(output);

//	auto aMean = a.mean(-1).unsqueeze(-1);

//	std::cout << "aOutput = " << aOutput << std::endl;
//	std::cout << "vOutput = " << vOutput << std::endl;
//	std::cout << "aMean = " << aMean << std::endl;

//	auto qOutput = a + (v + aMean);
//	std::cout << "vOutput + aMean = " << (vOutput + aMean) << std::endl;
//	std::cout << "qOutput = " << qOutput << std::endl;

	auto qOutput = v + (a - a.mean(-1, true));

	return qOutput;
}



