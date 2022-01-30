/*
 * airaccnnbmnet.cpp
 *
 *  Created on: Apr 29, 2021
 *      Author: zf
 */



#include "gymtest/cnnnets/airnets/airacbmnet.h"

#include <torch/torch.h>

AirACCnnBmNet::AirACCnnBmNet(int aNum):
	conv0(torch::nn::Conv2dOptions(4, 32, 8).stride(4)),
	bm0(32),
	conv1(torch::nn::Conv2dOptions(32, 64, 4).stride(2)),
	bm1(64),
	conv2(torch::nn::Conv2dOptions(64, 64, 3).stride(1)),
	bm2(64),
//	fc3(112896, 18)
	fc1(3136, 512),
	aOut(512, aNum),
	vOut(512, 1),
	actionNum(aNum)
//	fc3(1568, 18)
{
	register_module("conv0", conv0);
	register_module("bm0", bm0);
	register_module("conv1", conv1);
	register_module("bm1", bm1);
	register_module("conv2", conv2);
	register_module("bm2", bm2);
	register_module("fc1", fc1);
	register_module("aOut", aOut);
	register_module("vOut", vOut);
}

std::vector<torch::Tensor> AirACCnnBmNet::forward(torch::Tensor input) {
	torch::Tensor output = conv0->forward(input);
	output = bm0->forward(output);
	output = torch::leaky_relu(output);

	output = conv1->forward(output);
	output = bm1->forward(output);
	output = torch::leaky_relu(output);

	output = conv2->forward(output);
	output = bm2->forward(output);
	output = torch::leaky_relu(output);

//	std::cout << "fc input " << output.sizes() << std::endl;
	output = output.view({input.sizes()[0], -1});
	output = fc1->forward(output);
	output = torch::leaky_relu(output);
//	output = output.view({-1, 112896});
//	std::cout << "conv output sizes: " << output.sizes() << std::endl;


	auto aOutput = aOut->forward(output);
	auto vOutput = vOut->forward(output);

	return {aOutput, vOutput};
}
