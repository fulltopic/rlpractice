/*
 * testrawpolicy.cpp
 *
 *  Created on: Aug 20, 2021
 *      Author: zf
 */


#include <torch/torch.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

namespace {
const torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);

void testBinomial() {
	torch::Tensor count = torch::ones({1}) * 8;
	torch::Tensor prob = torch::ones({1}) * 0.5;

	torch::Tensor greedy = torch::binomial(count, prob);
	std::cout << "greedy: " << greedy << std::endl;
}

void testGt() {
	int count = 8;
	torch::Tensor probTensor = torch::rand({count});
	std::cout << "probs: " << probTensor << std::endl;

	torch::Tensor gtTensor = torch::gt(probTensor, 0.5);
	std::cout << "gt: " << gtTensor << std::endl;

	torch::Tensor valueTensor = torch::ones({count}, longOpt) * 2;
	torch::Tensor rcTensor = valueTensor * gtTensor;
	std::cout << "rc: " << rcTensor << std::endl;
}

void testRandAction() {
	torch::Tensor randActions = torch::randint(0, 8, {8}, longOpt);
	std::cout << "rand actions: " << randActions << std::endl;
}

void testArgsort() {
	int count = 8;
	int actionNum = 4;

	torch::Tensor probTensor = torch::rand({count, actionNum});
	torch::Tensor indice = probTensor.argsort(-1, true);

	std::cout << "prob: " << probTensor << std::endl;
	std::cout << "indice: " << indice << std::endl;

	torch::Tensor maxIndice = probTensor.argmax(-1);
	std::cout << "maxInice " << maxIndice << std::endl;
}

void getActions() {
	static const torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	int count = 8;
	int actionNum = 4;
	float epsilon = 0.5;

	torch::Tensor inputTensor = torch::rand({count, actionNum});
	std::cout << "input " << inputTensor << std::endl;
	auto batchSize = inputTensor.sizes()[0];

	torch::Tensor sampleTensor = torch::rand({batchSize});
	std::cout << "sample: " << sampleTensor << std::endl;
	torch::Tensor greedyTensor = torch::gt(sampleTensor, epsilon);
	std::cout << "greedy: " << greedyTensor << std::endl;

	torch::Tensor randActions = torch::randint(0, actionNum, {batchSize}, longOpt);
	std::cout << "rand: " << randActions << std::endl;
	torch::Tensor maxActions = inputTensor.argmax(-1);
	std::cout << "max: " << maxActions << std::endl;

	torch::Tensor actionTensor = randActions * greedyTensor.logical_not() + maxActions * greedyTensor;
	std::cout << "actionTensor: " << actionTensor << std::endl;

	std::vector<int64_t> actions;
	actions.assign(actionTensor.data_ptr<int64_t>(), actionTensor.data_ptr<int64_t>() + actionTensor.numel());
	std::cout << "actions: " << actions << std::endl;
}
}

int main() {
//	testBinomial();
//	testGt();
//	testRandAction();
//	testArgsort();

	getActions();
}

