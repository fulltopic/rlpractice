/*
 * testairenv.cpp
 *
 *  Created on: Apr 5, 2021
 *      Author: zf
 */

#include <gymtest/env/airenv.h>

#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

namespace
{
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("testgym"));
}
namespace {
//BeamRider: 9
void testGetInfo(std::string serverAddr) {
	const int clientNum = 2;
//	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_INFO(logger, "To connect to " << serverAddr);
//	AirEnv env(serverAddr, "SpaceInvaders-v0", clientNum);
//	AirEnv env(serverAddr, "Pong-v0", clientNum);
	//Qbert = 6
	//Pacman = 9
	AirEnv env(serverAddr, "BeamRiderNoFrameskip-v4", clientNum);

	auto info = env.init();
	auto actionSpace = std::get<1>(info);
	auto obSpace = std::get<0>(info);

    LOG4CXX_INFO(logger, "Action space: " << actionSpace.type << ", " << actionSpace.shape);
    LOG4CXX_INFO(logger, "Observation space:" << obSpace.type << "-" << obSpace.shape);

	auto rc = env.reset();
	LOG4CXX_INFO(logger, "next state: " << rc.size());

	auto actions = std::vector<long>(clientNum, 2);
	auto stepResult = env.step(1);
	auto obsvVec = std::get<0>(stepResult);
	auto rewardVec = std::get<1>(stepResult);
	auto doneVec = std::get<2>(stepResult);
	LOG4CXX_INFO(logger, "obsvVec: " << obsvVec.size());
	LOG4CXX_INFO(logger, "reward: " << rewardVec);
	LOG4CXX_INFO(logger, "done: " << doneVec);
}

void testEpisode() {
	const int clientNum = 2;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_INFO(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, "PongNoFrameskip-v4", clientNum);

	auto info = env.init();
	auto actionSpace = std::get<1>(info);
	auto obSpace = std::get<0>(info);

    LOG4CXX_INFO(logger, "Action space: " << actionSpace.type << ", " << actionSpace.shape);
    LOG4CXX_INFO(logger, "Observation space:" << obSpace.type << "-" << obSpace.shape);

    bool isDone = false;
    auto obsv = env.reset();
    while (!isDone) {
    	auto actions = std::vector<long>(clientNum, 3);
    	auto stepResult = env.step(actions, true);
    	obsv = std::get<0>(stepResult);
    	auto rewardVec = std::get<1>(stepResult);
    	auto doneVec = std::get<2>(stepResult);
    	LOG4CXX_INFO(logger, "reward: " << rewardVec);

    	isDone = false;
    	for (const auto &done: doneVec) {
    		if (done) {
    			isDone = true;
    			break;
    		}
    	}
    }
}

void testReset() {
	const int clientNum = 2;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_INFO(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, "Alien-v0", clientNum);

	auto info = env.init();
	auto actionSpace = std::get<1>(info);
	auto obSpace = std::get<0>(info);

    LOG4CXX_INFO(logger, "Action space: " << actionSpace.type << ", " << actionSpace.shape);
    LOG4CXX_INFO(logger, "Observation space:" << obSpace.type << "-" << obSpace.shape);

    bool isDone = false;
    auto obsv = env.reset();
    LOG4CXX_INFO(logger, "reset " << obsv.size());
    while (!isDone) {
    	auto actions = std::vector<long>(clientNum, 3);
    	auto stepResult = env.step(actions);
    	obsv = std::get<0>(stepResult);
    	auto rewardVec = std::get<1>(stepResult);
    	auto doneVec = std::get<2>(stepResult);
//    	LOG4CXX_INFO(logger, "reward: " << rewardVec);

    	for (int i = 0; i < doneVec.size(); i ++) {
    		if (doneVec[i]) {
    			auto tmpObs = env.reset(i);
    			LOG4CXX_INFO(logger, "Reset client " << i << "result: " << tmpObs.size());
    		}
    	}
    }
}

}

int main(int argc, char** argv) {
	log4cxx::BasicConfigurator::configure();

	testGetInfo(argv[1]);
//	testEpisode();
//	testReset();

	LOG4CXX_INFO(logger, "End of test");
}





