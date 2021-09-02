/*
 * noisycartfcnet.cpp
 *
 *  Created on: Aug 31, 2021
 *      Author: zf
 */


#include <torch/torch.h>

#include "gymtest/noisynets/noisycartfcnet.h"


NoisyCartFcNet::NoisyCartFcNet(int input, int output):
	inputNum(input),
	outputNum(output),
	fc0(input, 256),
	fcOutput(std::make_shared<NoisyLinear>(256, output))
{
	register_module("fc0", fc0);
//	register_module("fcOutput", fcOutput);
	fcOutput = register_module("fcOutput", fcOutput);

//	init_weights(fc0->named_parameters(), sqrt(2.0), 0);
//	init_weights(fcOutput->named_parameters(), sqrt(2.0), 0);
}

torch::Tensor NoisyCartFcNet::forward(torch::Tensor input) {
	torch::Tensor output = fc0->forward(input);
	output = torch::relu(output);
	output = fcOutput->forward(output);
//	output = torch::relu(output);

	return output;
}

void NoisyCartFcNet::resetNoise() {
	fcOutput->resetNoise();
}
