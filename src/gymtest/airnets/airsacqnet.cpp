/*
 * airsacqnet.cpp
 *
 *  Created on: Sep 21, 2021
 *      Author: zf
 */




#include <torch/torch.h>

#include <iostream>

#include "gymtest/airnets/airsacqnet.h"
#include "gymtest/utils/netinitutils.h"

AirSacQNet::AirSacQNet(const int outputNum):
	conv0(torch::nn::Conv2dOptions(4, 32, 8).stride(4)),
	conv1(torch::nn::Conv2dOptions(32, 64, 4).stride(2)),
	conv2(torch::nn::Conv2dOptions(64, 64, 3).stride(1)),
//	fc3(112896, 18)
	fc2(3136, 512),
	fc3(512, outputNum)
//	fc3(1568, 18)
{
	register_module("conv0", conv0);
	register_module("conv1", conv1);
	register_module("conv2", conv2);
	register_module("fc2", fc2);
	register_module("fc3", fc3);


	NetInitUtils::Init_weights(conv0->named_parameters(), sqrt(2.0), 0, NetInitUtils::Kaiming);
	NetInitUtils::Init_weights(conv1->named_parameters(), sqrt(2.0), 0, NetInitUtils::Kaiming);
	NetInitUtils::Init_weights(conv2->named_parameters(), sqrt(2.0), 0, NetInitUtils::Kaiming);
//	NetInitUtils::Init_weights(fc2->named_parameters(), sqrt(2.0), 0, NetInitUtils::Kaiming);
//	NetInitUtils::Init_weights(fc3->named_parameters(), sqrt(2.0), 0, NetInitUtils::Kaiming);
}

torch::Tensor AirSacQNet::forward(torch::Tensor input) {
	torch::Tensor output = conv0->forward(input);
	output = torch::relu(output);

	output = conv1->forward(output);
	output = torch::relu(output);

	output = conv2->forward(output);
	output = torch::relu(output);

//	std::cout << "fc input " << output.sizes() << std::endl;
	output = output.view({input.sizes()[0], -1});

	output = fc2->forward(output);
//	std::cout << "fc2 output" << std::endl;
	output = torch::relu(output);
//	output = output.view({-1, 112896});
//	std::cout << "conv output sizes: " << output.sizes() << std::endl;

	output = fc3->forward(output);

	return output;
}
