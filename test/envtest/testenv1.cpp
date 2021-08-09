/*
 * testenv1.cpp
 *
 *  Created on: Aug 7, 2021
 *      Author: zf
 */


#include "probeenvs/ProbeEnvWrapper.h"
#include <string>
#include <iostream>
#include <vector>
#include <random>
//#include <log4cxx/logger.h>
//#include <log4cxx/basicconfigurator.h>
//#include <log4cxx/consoleappender.h>
//#include <log4cxx/simplelayout.h>
//#include <log4cxx/logmanager.h>

namespace {
//log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("testenv1"));

template<typename T>
void printVec(const std::string& name, const std::vector<T>& data) {
	std::cout << name << ": " << std::endl;
	for (int i = 0; i < data.size(); i ++) {
		std::cout << data[i] << ", ";
	}
	std::cout << std::endl;
}

void test0() {
	auto env = ProbeVecEnv::CreateEnv(8, 1, 2);

	auto resetRc = env->reset();
	printVec("reset state", resetRc);

	auto actions = std::vector<long>(2, 1);

	auto rc = env->step(actions);

	auto state = std::get<0>(rc);
	auto reward = std::get<1>(rc);
	auto done = std::get<2>(rc);

	printVec("state", state);
	printVec("reward", reward);
	printVec("done", done);
}

//TODO: action number
void test1(int inputLen, int envId, int envNum, int actionNum) {
	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution = std::uniform_int_distribution<int>(0, actionNum - 1);

	ProbeEnvWrapper env(inputLen, envId, envNum);

	std::cout << "---------------------------------------> reset " << std::endl;
	auto resetRc = env.reset();
	printVec("reset rc", resetRc);

	for (int i = 0; i < 8; i ++) {
		std::cout << "-------------------------------------> step " << i << std::endl;

		std::vector<long> actions(envNum, 0);
		for (int j = 0; j < actions.size(); j ++) {
			actions[j] = distribution(generator);
		}
		printVec("actions", actions);
		auto rc = env.step(actions);

		auto state = std::get<0>(rc);
		auto reward = std::get<1>(rc);
		auto done = std::get<2>(rc);

		printVec("state", state);
		printVec("reward", reward);
		printVec("done", done);
	}
}
}

int main(int argc, char** argv) {
	test1(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
}
