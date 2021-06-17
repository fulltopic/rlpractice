/*
 * testsimplenet.cpp
 *
 *  Created on: Apr 4, 2021
 *      Author: zf
 */

#include "gymtest/env/lunarenv.h"

#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>
#include "gymtest/lunarnets/lunarfcnet.h"

namespace
{
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("testsimplenet"));

void testSimpleNet() {
	std::string envName = "LunarLander-v2";
	torch::Device deviceType = torch::kCUDA;
	LunarFcNet model;
	model.to(deviceType);

	const int clientNum = 2;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_INFO(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);

	auto info = env.init();
	auto actionSpace = std::get<1>(info);
	auto obSpace = std::get<0>(info);

	LOG4CXX_INFO(logger, "Get actionSpace " << actionSpace.type << ": " << actionSpace.shape);
	LOG4CXX_INFO(logger, "Get observeSpace " << obSpace.type << ": " << obSpace.shape);

	auto stateVec = env.reset();
	torch::Tensor inputTensor = torch::from_blob(stateVec.data(), {clientNum, NetConfig::LunarInputW}).to(deviceType);
	torch::Tensor outputTensor = model.forward(inputTensor).to(torch::kCPU);
	torch::Tensor actionTensor = outputTensor.argmax(-1);
	std::vector<long> actionVec(actionTensor.data_ptr<long>(), actionTensor.data_ptr<long>() + actionTensor.numel());
	LOG4CXX_INFO(logger, "actionTensor: " << actionTensor);
	LOG4CXX_INFO(logger, "actionVec: " << actionVec);

	auto result = env.step(actionVec);
	inputTensor = torch::from_blob(std::get<0>(result).data(), {clientNum, NetConfig::LunarInputW}).to(deviceType);
	outputTensor = model.forward(inputTensor).to(torch::kCPU);
	actionTensor = outputTensor.argmax(-1);
	actionVec = std::vector<long>(actionTensor.data_ptr<long>(), actionTensor.data_ptr<long>() + actionTensor.numel());
	LOG4CXX_INFO(logger, "actionTensor: " << actionTensor);
	LOG4CXX_INFO(logger, "actionVec: " << actionVec);
}

void testEpisode() {
	std::string envName = "LunarLander-v2";

	torch::Device deviceType = torch::kCUDA;
	LunarFcNet model;
	model.to(deviceType);

	const int clientNum = 2;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_INFO(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);

	auto info = env.init();

	auto stateVec = env.reset();
	torch::Tensor inputTensor = torch::from_blob(stateVec.data(), {clientNum, NetConfig::LunarInputW}).to(deviceType);

	bool isRunning = true;
	while (isRunning) {
		auto outputTensor = model.forward(inputTensor).to(torch::kCPU);
		auto actionTensor = outputTensor.argmax(-1);
		std::vector<long> actionVec(actionTensor.data_ptr<long>(), actionTensor.data_ptr<long>() + actionTensor.numel());
		LOG4CXX_INFO(logger, "actionTensor: " << actionTensor);
		LOG4CXX_INFO(logger, "actionVec: " << actionVec);

		auto result = env.step(actionVec);
		inputTensor = torch::from_blob(std::get<0>(result).data(), {clientNum, NetConfig::LunarInputW}).to(deviceType);
		LOG4CXX_INFO(logger, "reward: " << std::get<1>(result));

		auto doneVec = std::get<2>(result);
		for (auto isDone: doneVec) {
			if (isDone) {
				isRunning = false;
			}
		}
	}

}
}

int main() {
	log4cxx::BasicConfigurator::configure();

	testSimpleNet();
//	testEpisode();

	LOG4CXX_INFO(logger, "End of test");
}
