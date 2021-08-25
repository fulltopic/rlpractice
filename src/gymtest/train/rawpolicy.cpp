/*
 * rawpolicy.cpp
 *
 *  Created on: Apr 10, 2021
 *      Author: zf
 */



#include "gymtest/train/rawpolicy.h"
#include <vector>

RawPolicy::RawPolicy(float ep, int an): epsilon(ep), actionNum(an),
	logger(log4cxx::Logger::getLogger("rawpolicy"))
{
	intOpt = intOpt.dtype(torch::kInt);
}

//std::vector<int64_t> RawPolicy::getActions(torch::Tensor input) {
////	static float count = 0;
////	static float ave = 0;
//	auto batchSize = input.sizes()[0];
//	auto randTensor = torch::rand({batchSize});
//	float* randValue = randTensor.data_ptr<float>();
//	auto randActionTensor = torch::randint(0, actionNum, randTensor.sizes(), intOpt);
//	int32_t* randActions = randActionTensor.data_ptr<int32_t>();
//
//	{
//		torch::NoGradGuard gd;
//		auto actionTensor = input.argmax(-1).to(torch::kCPU);
////	LOG4CXX_INFO(logger, "actionTensor: " << actionTensor);
////	LOG4CXX_INFO(logger, "input: " << input);
//		std::vector<int64_t> actionData(actionTensor.data_ptr<int64_t>(), actionTensor.data_ptr<int64_t>() + batchSize);
////	int diff = 0;
//		for (int i = 0; i < batchSize; i ++) {
//			if (randValue[i] <= epsilon) {
//				actionData[i] = randActions[i];
////			diff ++;
//			}
//		}
////	count ++;
////	ave += (float)(diff - ave) / (count);
////	LOG4CXX_INFO(logger, "diff of rand actions: " << diff << ", " << ave);
//		return actionData;
//	}
//}

//std::vector<int64_t> RawPolicy::getActions(torch::Tensor input) {
//	torch::NoGradGuard gd;
//
//	torch::Tensor inputTensor = input.to(torch::kCPU);
//	auto batchSize = inputTensor.sizes()[0];
//	const torch::Tensor probTensor(torch::ones({1}) * epsilon);
//	static const torch::Tensor countTensor = torch::ones({1});
//	static const torch::Tensor tieTensor (torch::ones({1}) * 0.5);
//	std::vector<int64_t> actions(batchSize, 0);
//
//	for (int i = 0; i < batchSize; i ++) {
//		torch::Tensor greedyTensor = torch::binomial(countTensor, probTensor);
//		int greedyValue = greedyTensor.item().toInt();
//
//		if (greedyValue == 1) {
//			actions[i] = (int)(torch::rand({1}).item<float>() * actionNum);
//		} else {
//			auto data = inputTensor[i].data_ptr<float>();
//			float maxValue = data[0];
//			std::vector<int> indices;
//			for (int j = 0; j < actionNum; j ++) {
//				if (data[j] > maxValue) {
//					maxValue = data[j];
//					indices.clear();
//					indices.push_back(j);
//				} else if (data[j] == maxValue) {
//					indices.push_back(j);
//				}
//			}
//
//			if(indices.size() == 1) {
//				actions[i] = indices[0];
//			} else {
//				int index = (int)(torch::rand({1}).item<float>() * indices.size());
//				actions[i] = indices[index];
//			}
//		}
//	}
//
//	return actions;
//}

std::vector<int64_t> RawPolicy::getActions(torch::Tensor input) {
	static const torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);

	torch::NoGradGuard guard;

	torch::Tensor inputTensor = input.to(torch::kCPU);
	auto batchSize = inputTensor.sizes()[0];

	torch::Tensor sampleTensor = torch::rand({batchSize});
	torch::Tensor greedyTensor = torch::gt(sampleTensor, epsilon);

	torch::Tensor randActions = torch::randint(0, actionNum, {batchSize}, longOpt);
	torch::Tensor maxActions = inputTensor.argmax(-1);

	torch::Tensor actionTensor = randActions * greedyTensor.logical_not() + maxActions * greedyTensor;

	std::vector<int64_t> actions;
	actions.assign(actionTensor.data_ptr<int64_t>(), actionTensor.data_ptr<int64_t>() + actionTensor.numel());
	return actions;
}

std::vector<int64_t> RawPolicy::getTestActions(torch::Tensor input) {
	torch::NoGradGuard gd;
	auto batchSize = input.sizes()[0];
//	LOG4CXX_INFO(logger, "Policy input: " << input);

	auto actionTensor = input.argmax(-1).to(torch::kCPU);
//	LOG4CXX_INFO(logger, "actionTensor: " << actionTensor);
	std::vector<int64_t> actionData;
	actionData.assign(actionTensor.data_ptr<int64_t>(), actionTensor.data_ptr<int64_t>() + actionTensor.numel());
//	LOG4CXX_INFO(logger, "actionTensor " << actionTensor);

	//vector copied
	return actionData;
}

void RawPolicy::updateEpsilon(float e) {
	epsilon = e;
}

float RawPolicy::getEpsilon() {
	return epsilon;
}
