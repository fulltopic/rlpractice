/*
 * airsmallkernelnet.cpp
 *
 *  Created on: May 25, 2021
 *      Author: zf
 */




#include "gymtest/airnets/airsmallkernelnet.h"

#include "gymtest/utils/netinitutils.h"

#include <torch/torch.h>

#include <algorithm>
#include <string.h>

AirSKCnnNet::AirSKCnnNet(int aNum):
	conv0(torch::nn::Conv2dOptions(4, 32, 5).stride(2)),
	conv1(torch::nn::Conv2dOptions(32, 64, 5).stride(2)),
	conv2(torch::nn::Conv2dOptions(64, 128, 5).stride(1)),
	conv3(torch::nn::Conv2dOptions(128, 256, 3).stride(1)),
	fc1(36864, 512),
	aOut(512, aNum),
	actionNum(aNum)
{
	register_module("conv0", conv0);
	register_module("conv1", conv1);
	register_module("conv2", conv2);
	register_module("conv3", conv3);
	register_module("fc1", fc1);
	register_module("aOut", aOut);

//	NetInitUtils::Init_weights(conv0->named_parameters(), sqrt(2.0), 0);
//	NetInitUtils::Init_weights(conv1->named_parameters(), sqrt(2.0), 0);
//	NetInitUtils::Init_weights(conv2->named_parameters(), sqrt(2.0), 0);
//	NetInitUtils::Init_weights(conv3->named_parameters(), sqrt(2.0), 0);
//	NetInitUtils::Init_weights(fc1->named_parameters(), sqrt(2.0), 0);
//	NetInitUtils::Init_weights(aOut->named_parameters(), sqrt(2.0), 0);
//	NetInitUtils::Init_weights(vOut->named_parameters(), sqrt(2.0), 0);

//	torch::nn::init::kaiming_uniform_(conv0->weight);
}

torch::Tensor AirSKCnnNet::forward(torch::Tensor input) {
	torch::Tensor output = conv0->forward(input);
	output = torch::leaky_relu(output);

	output = conv1->forward(output);
	output = torch::leaky_relu(output);

	output = conv2->forward(output);
	output = torch::leaky_relu(output);

	output = conv3->forward(output);
	output = torch::leaky_relu(output);

//	std::cout << "fc input " << output.sizes() << std::endl;
	output = output.view({input.sizes()[0], -1});
	output = fc1->forward(output);
	output = torch::leaky_relu(output);


	auto aOutput = aOut->forward(output);

	return aOutput;
}



