/*
 * airdueling.cpp
 *
 *  Created on: Apr 21, 2021
 *      Author: zf
 */


#include <torch/torch.h>

#include <iostream>

#include "gymtest/airnets/airdueling.h"

AirCnnDuelNet::AirCnnDuelNet(const int outputNum):
	conv0(torch::nn::Conv2dOptions(4, 32, 8).stride(4)),
	conv1(torch::nn::Conv2dOptions(32, 64, 4).stride(2)),
	conv2(torch::nn::Conv2dOptions(64, 64, 3).stride(1)),
	vfc0(3136, 512),
	vfc1(512, 1),
	afc0(3136, 512),
	afc1(512, outputNum),
	actionNum(outputNum)
{
	register_module("conv0", conv0);
	register_module("conv1", conv1);
	register_module("conv2", conv2);
	register_module("vfc0", vfc0);
	register_module("vfc1", vfc1);
	register_module("afc0", afc0);
	register_module("afc1", afc1);
}

torch::Tensor AirCnnDuelNet::forward(torch::Tensor input) {
	torch::Tensor output = conv0->forward(input);
	output = torch::relu(output);

	output = conv1->forward(output);
	output = torch::relu(output);

	output = conv2->forward(output);
	output = torch::relu(output);

//	std::cout << "fc input " << output.sizes() << std::endl;
	output = output.view({input.sizes()[0], -1});


	auto vOutput = vfc0->forward(output);
	vOutput = torch::relu(vOutput);
	vOutput = vfc1->forward(vOutput);

	auto aOutput = afc0->forward(output);
	aOutput = torch::relu(aOutput);
	aOutput = afc1->forward(aOutput);
//	auto aMean = aOutput.mean(-1).unsqueeze(-1);

//	std::cout << "aOutput = " << aOutput << std::endl;
//	std::cout << "vOutput = " << vOutput << std::endl;
//	std::cout << "aMean = " << aMean << std::endl;

//	auto qOutput = aOutput + (vOutput + aMean);
//	std::cout << "vOutput + aMean = " << (vOutput + aMean) << std::endl;
//	std::cout << "qOutput = " << qOutput << std::endl;

	auto qOutput = vOutput + (aOutput - aOutput.mean(-1, true));

	return qOutput;
}

