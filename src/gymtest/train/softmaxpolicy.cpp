/*
 * softmaxpolicy.cpp
 *
 *  Created on: Apr 30, 2021
 *      Author: zf
 */


#include "gymtest/train/softmaxpolicy.h"

SoftmaxPolicy::SoftmaxPolicy(int an): actionNum(an)
{
	gen = std::mt19937(rd());
	intOpt = intOpt.dtype(torch::kInt);
}

//Input is softmax output
std::vector<int64_t> SoftmaxPolicy::getActions(torch::Tensor input) {
	auto batchSize = input.sizes()[0];
	std::vector<int64_t> actions(batchSize, 0);
	//TODO: GPU parallel
//	{
////		torch::NoGradGuard gd;
//		torch::Tensor distParam = input.to(torch::kCPU);
//		auto offset = distParam.numel() / batchSize;
//		for (int i = 0; i < batchSize; i ++) {
//			std::vector<float> param(distParam.data_ptr<float>() + offset * i, distParam.data_ptr<float>() + offset * (i + 1));
//			std::discrete_distribution<int> d(param.begin(), param.end());
//			actions[i] = d(gen);
//		}
//	}
	{
//		LOG4CXX_INFO(logger, "input shape: " << input.sizes());
		torch::NoGradGuard guard;
		auto samples = input.multinomial(1).to(torch::kCPU);
//		LOG4CXX_INFO(logger, "samples: " << samples);
		long* samplePtr = samples.data_ptr<long>();
		for (int i = 0; i < actions.size(); i ++) {
			actions[i] = samplePtr[i];
		}
	}

	return actions;
}

std::vector<int64_t> SoftmaxPolicy::getTestActions(torch::Tensor input) {
	torch::NoGradGuard gd;
	auto batchSize = input.sizes()[0];

	auto actionTensor = input.argmax(-1).to(torch::kCPU);
	std::vector<int64_t> actionData(actionTensor.data_ptr<int64_t>(), actionTensor.data_ptr<int64_t>() + batchSize);
//	LOG4CXX_INFO(logger, "actionTensor " << actionTensor);

	//vector copied
	return actionData;
}
