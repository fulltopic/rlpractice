/*
 * testaircnn.cpp
 *
 *  Created on: Apr 5, 2021
 *      Author: zf
 */

#include <torch/torch.h>

#include <vector>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include "gymtest/airnets/aircnnnet.h"
#include "gymtest/env/airenv.h"

namespace {
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("testaircnn"));

void testNetStruct() {
	const int clientNum = 2;
	torch::Device deviceType = torch::kCUDA;

	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_INFO(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, "Alien-v0", clientNum);

	auto info = env.init();
	auto actionSpace = std::get<1>(info);
	auto obSpace = std::get<0>(info);

    LOG4CXX_INFO(logger, "Action space: " << actionSpace.type << ", " << actionSpace.shape);
    LOG4CXX_INFO(logger, "Observation space:" << obSpace.type << "-" << obSpace.shape);

    AirCnnNet model(18);
    model.to(deviceType);
    auto obsvVec = env.reset();
    torch::Tensor input = torch::from_blob(obsvVec.data(), {clientNum, 4, 84, 84}).to(deviceType);
    torch::Tensor outputTensor = model.forward(input);
    auto actionTensor = outputTensor.argmax(-1).to(torch::kCPU);
    std::vector<long> actions(actionTensor.data_ptr<long>(), actionTensor.data_ptr<long>() + actionTensor.numel());

    auto result = env.step(actions);
}

void testEpisode(int num) {
	const int clientNum = num;
	torch::Device deviceType = torch::kCUDA;

	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_INFO(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, "Alien-v0", clientNum);

	auto info = env.init();
	auto actionSpace = std::get<1>(info);
	auto obSpace = std::get<0>(info);

    LOG4CXX_INFO(logger, "Action space: " << actionSpace.type << ", " << actionSpace.shape);
    LOG4CXX_INFO(logger, "Observation space:" << obSpace.type << "-" << obSpace.shape);

    AirCnnNet model(18);
    model.to(deviceType);

    bool isDone = false;
    auto obsvVec = env.reset();

    while (!isDone) {
    	torch::Tensor input = torch::from_blob(obsvVec.data(), {clientNum, 4, 84, 84}).to(deviceType);
    	torch::Tensor outputTensor = model.forward(input);
    	auto actionTensor = outputTensor.argmax(-1).to(torch::kCPU);
    	std::vector<long> actions(actionTensor.data_ptr<long>(), actionTensor.data_ptr<long>() + actionTensor.numel());

    	auto result = env.step(actions);
    	obsvVec = std::get<0>(result);
    	auto rewardVec = std::get<1>(result);
    	LOG4CXX_INFO(logger, "Rewards: " << rewardVec);
    	auto doneVec = std::get<2>(result);
    	for (const auto& done: doneVec) {
    		if (done) {
    			isDone = true;
    		}
    	}
    }
}

void testOutput() {
	AirCnnNet model(18);
	model.to(torch::kCUDA);
	model.train();

	auto input = torch::rand({32, 4, 84, 84}).to(torch::kCUDA);

	auto output0 = model.forward(input);
	LOG4CXX_INFO(logger, "output0: " << output0);
	auto action0 = output0.argmax(-1);
	LOG4CXX_INFO(logger, "action0: " << action0);

	model.eval();
	auto output1 = model.forward(input);
	LOG4CXX_INFO(logger, "output1: " << output1);
	LOG4CXX_INFO(logger, "equals -----------------------> " << output1.equal(output0));
	auto action1 = output1.argmax(-1);
	LOG4CXX_INFO(logger, "action1: " << action1);
}
}

int main(int argc, char** argv) {
	log4cxx::BasicConfigurator::configure();

//	testNetStruct();
//	testEpisode(atoi(argv[1]));
	testOutput();

	LOG4CXX_INFO(logger, "End of test");
	return 0;
}



