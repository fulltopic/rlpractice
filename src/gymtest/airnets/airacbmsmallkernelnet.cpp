/*
 * airacbmsmallkernelnet.cpp
 *
 *  Created on: May 7, 2021
 *      Author: zf
 */



#include "gymtest/airnets/airacbmsmallkernelnet.h"

#include "gymtest/utils/netinitutils.h"

#include <torch/torch.h>

#include <algorithm>
#include <string.h>

AirACSKCnnBmNet::AirACSKCnnBmNet(int aNum):
	conv0(torch::nn::Conv2dOptions(4, 32, 5).stride(2)),
//	bm0(32),
	conv1(torch::nn::Conv2dOptions(32, 64, 5).stride(2)),
//	bm1(64),
	conv2(torch::nn::Conv2dOptions(64, 128, 5).stride(1)),
//	bm2(128),
	conv3(torch::nn::Conv2dOptions(128, 256, 3).stride(1)),
//	bm3(256),
//	fc3(112896, 18)
	fc1(36864, 512),
	aOut(512, aNum),
	vOut(512, 1),
	actionNum(aNum)
//	fc3(1568, 18)
{
	register_module("conv0", conv0);
//	register_module("bm0", bm0);
	register_module("conv1", conv1);
//	register_module("bm1", bm1);
	register_module("conv2", conv2);
//	register_module("bm2", bm2);
	register_module("conv3", conv3);
//	register_module("bm3", bm3);
	register_module("fc1", fc1);
	register_module("aOut", aOut);
	register_module("vOut", vOut);

//	NetInitUtils::Init_weights(conv0->named_parameters(), sqrt(2.0), 0);
//	NetInitUtils::Init_weights(conv1->named_parameters(), sqrt(2.0), 0);
//	NetInitUtils::Init_weights(conv2->named_parameters(), sqrt(2.0), 0);
//	NetInitUtils::Init_weights(conv3->named_parameters(), sqrt(2.0), 0);
//	NetInitUtils::Init_weights(fc1->named_parameters(), sqrt(2.0), 0);
//	NetInitUtils::Init_weights(aOut->named_parameters(), sqrt(2.0), 0);
//	NetInitUtils::Init_weights(vOut->named_parameters(), sqrt(2.0), 0);

//	torch::nn::init::kaiming_uniform_(conv0->weight);
}

std::vector<torch::Tensor> AirACSKCnnBmNet::forward(torch::Tensor input) {
	torch::Tensor output = conv0->forward(input);
//	output = bm0->forward(output);
	output = torch::leaky_relu(output);

	output = conv1->forward(output);
//	output = bm1->forward(output);
	output = torch::leaky_relu(output);

	output = conv2->forward(output);
//	output = bm2->forward(output);
	output = torch::leaky_relu(output);

	output = conv3->forward(output);
//	output = bm3->forward(output);
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



