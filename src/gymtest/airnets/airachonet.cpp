/*
 * airachonet.cpp
 *
 *  Created on: Jun 4, 2021
 *      Author: zf
 */



#include "gymtest/airnets/airachonet.h"
#include "gymtest/utils/netinitutils.h"

#include <torch/torch.h>

AirACHONet::AirACHONet(int aNum):
	conv0(torch::nn::Conv2dOptions(4, 32, 8).stride(4)),
	conv1(torch::nn::Conv2dOptions(32, 64, 4).stride(2)),
	conv2(torch::nn::Conv2dOptions(64, 64, 3).stride(1)),
//	fc3(112896, 18)
	afc(3136, 512),
	vfc(3136, 512),
	aOut(512, aNum),
	vOut(512, 1),
	actionNum(aNum)
//	fc3(1568, 18)
{
	register_module("conv0", conv0);
	register_module("conv1", conv1);
	register_module("conv2", conv2);
	register_module("afc", afc);
	register_module("vfc", vfc);
	register_module("aOut", aOut);
	register_module("vOut", vOut);

	NetInitUtils::Init_weights(conv0->named_parameters(), sqrt(2.0), 0);
	NetInitUtils::Init_weights(conv1->named_parameters(), sqrt(2.0), 0);
	NetInitUtils::Init_weights(conv2->named_parameters(), sqrt(2.0), 0);
	NetInitUtils::Init_weights(afc->named_parameters(), sqrt(2.0), 0);
	NetInitUtils::Init_weights(vfc->named_parameters(), sqrt(2.0), 0);
	NetInitUtils::Init_weights(aOut->named_parameters(), sqrt(2.0), 0);
	NetInitUtils::Init_weights(vOut->named_parameters(), sqrt(2.0), 0);
}

std::vector<torch::Tensor> AirACHONet::forward(torch::Tensor input) {
	torch::Tensor output = conv0->forward(input);
	output = torch::relu(output);

	output = conv1->forward(output);
	output = torch::relu(output);

	output = conv2->forward(output);
	output = torch::relu(output);

//	std::cout << "fc input " << output.sizes() << std::endl;
	output = output.view({input.sizes()[0], -1});

	auto aOutput = afc->forward(output);
	aOutput = torch::relu(aOutput);

	auto vOutput = vfc->forward(output);
	vOutput = torch::relu(vOutput);
//	output = output.view({-1, 112896});
//	std::cout << "conv output sizes: " << output.sizes() << std::endl;


	aOutput = aOut->forward(aOutput);
	vOutput = vOut->forward(vOutput);

	return {aOutput, vOutput};
}
