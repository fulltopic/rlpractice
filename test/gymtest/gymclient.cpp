/*
 * gymclient.cpp
 *
 *  Created on: Apr 2, 2021
 *      Author: zf
 */

#include <gymtest/env/lunarenv.h>

#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

namespace
{
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("testgym"));
}
namespace {
void testGetInfo() {
	std::string envName = "LunarLander-v2";

	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_INFO(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName);

	auto info = env.init();
	auto actionSpace = std::get<1>(info);
	auto obSpace = std::get<0>(info);

	LOG4CXX_INFO(logger, "Get actionSpace " << actionSpace.type);
	LOG4CXX_INFO(logger, "Get observeSpace " << obSpace.type);

	auto rc = env.reset();
	LOG4CXX_INFO(logger, "next state: " << rc);
//	LOG4CXX_INFO(logger, "next state: " << std::get<0>(rc));
//	LOG4CXX_INFO(logger, "reward: " << std::get<1>(rc));
//	LOG4CXX_INFO(logger, "Done: " << std::get<2>(rc));

	bool isDone = false;
	while (!isDone) {
		auto result = env.step(2);
		auto reward = std::get<1>(result);
		isDone = std::get<2>(result);

		LOG4CXX_INFO(logger, "Reward: " << reward);
	}

	env.reset();
	LOG4CXX_INFO(logger, "Reset");
	auto result = env.step(1);
	LOG4CXX_INFO(logger, "reward: " << std::get<1>(result));
}

void testMultiClient() {
	std::string envName = "LunarLander-v2";

	const int clientNum = 2;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_INFO(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);

	auto info = env.init();
	auto actionSpace = std::get<1>(info);
	auto obSpace = std::get<0>(info);

	LOG4CXX_INFO(logger, "Get actionSpace " << actionSpace.type << ": " << actionSpace.shape);
	LOG4CXX_INFO(logger, "Get observeSpace " << obSpace.type << ": " << obSpace.shape);

//	bool done = false;
//	std::vector<long> actions(clientNum, 1);
//	do {
//		auto result = env.step(actions);
//		auto rewards = std::get<1>(result);
//		auto dones = std::get<2>(result);
//
//		LOG4CXX_INFO(logger, "Get rewards: " << rewards);
//
//		done = true;
//		for (auto isDone: dones) {
//			done = done && isDone;
//		}
//	} while (!done);

}
}

int main() {
	log4cxx::BasicConfigurator::configure();

	testGetInfo();
//	testMultiClient();

	LOG4CXX_INFO(logger, "End of test");
}



