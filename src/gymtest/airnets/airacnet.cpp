/*
 * airacnet.cpp
 *
 *  Created on: May 4, 2021
 *      Author: zf
 */




/*
 * airaccnnbmnet.cpp
 *
 *  Created on: Apr 29, 2021
 *      Author: zf
 */



#include "gymtest/airnets/airacnet.h"
#include "gymtest/utils/netinitutils.h"

#include <torch/torch.h>

AirACCnnNet::AirACCnnNet(int aNum):
	conv0(torch::nn::Conv2dOptions(4, 32, 8).stride(4)),
	conv1(torch::nn::Conv2dOptions(32, 64, 4).stride(2)),
	conv2(torch::nn::Conv2dOptions(64, 64, 3).stride(1)),
//	fc3(112896, 18)
	fc1(3136, 512),
	aOut(512, aNum),
	vOut(512, 1),
	actionNum(aNum)
//	fc3(1568, 18)
{
	register_module("conv0", conv0);
	register_module("conv1", conv1);
	register_module("conv2", conv2);
	register_module("fc1", fc1);
	register_module("aOut", aOut);
	register_module("vOut", vOut);

	NetInitUtils::Init_weights(conv0->named_parameters(), sqrt(2.0), 0);
	NetInitUtils::Init_weights(conv1->named_parameters(), sqrt(2.0), 0);
	NetInitUtils::Init_weights(conv2->named_parameters(), sqrt(2.0), 0);
	NetInitUtils::Init_weights(fc1->named_parameters(), sqrt(2.0), 0);
	NetInitUtils::Init_weights(aOut->named_parameters(), sqrt(2.0), 0);
	NetInitUtils::Init_weights(vOut->named_parameters(), sqrt(2.0), 0);
}

std::vector<torch::Tensor> AirACCnnNet::forward(torch::Tensor input) {
	torch::Tensor output = conv0->forward(input);
	output = torch::leaky_relu(output);

	output = conv1->forward(output);
	output = torch::leaky_relu(output);

	output = conv2->forward(output);
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
