/*
 * testmodelsync.cpp
 *
 *  Created on: Oct 15, 2021
 *      Author: zf
 */


#include "gymtest/airnets/aircnnnet.h"
#include "gymtest/airnets/airsacqnet.h"
#include "gymtest/airnets/airsacpnet.h"

#include <torch/torch.h>

#include <iostream>
#include <string>

namespace {
void syncModel(AirSacQNet& net, AirSacQNet& target) {
	static const float tau = 1;

	torch::NoGradGuard guard;

	auto paramDict1 = net.named_parameters();
	auto buffDict1 = net.named_buffers();
	auto targetParamDict1 = target.named_parameters();
	auto targetBuffDict1 = target.named_buffers();

	for (const auto& item: paramDict1) {
		const auto& key = item.key();
		const auto param = item.value();
		auto& targetParam = targetParamDict1[key];

		targetParam.mul_(1 - tau);
		targetParam.add_(param, tau);
	}

	for (const auto& item: buffDict1) {
		const auto& key = item.key();
		const auto& buff = item.value();
		auto& targetBuff = targetBuffDict1[key];

		targetBuff.mul(1 - tau);
		targetBuff.add_(buff, tau);
	}
	std::cout << "target network 1 synched" << std::endl;
}

void step(AirSacQNet& net, AirSacQNet& targetNet, torch::Tensor input) {
	torch::Tensor output = net.forward(input);
	torch::Tensor tOutput = targetNet.forward(input);
	std::cout << "Output: " << output << std::endl;
	std::cout << "Target output: " << tOutput << std::endl;
}

void test0() {
	int batch = 4;
	int outputNum = 4;

	AirSacQNet net(outputNum);
	AirSacQNet targetNet(outputNum);
	torch::Tensor input = torch::rand({batch, 4, 84, 84});

	std::cout << "Before sync: " << std::endl;
	step(net, targetNet, input);

	syncModel(net, targetNet);
	std::cout << "After sync: " << std::endl;
	step(net, targetNet, input);
}
}

int main(int argc, char** argv) {
	test0();

	return 0;
}

