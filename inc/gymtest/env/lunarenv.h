/*
 * lunarenv.h
 *
 *  Created on: Apr 2, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_LUNARENV_H_
#define INC_GYMTEST_LUNARENV_H_

#include <string>
#include <vector>

#include <cpprl/cpprl.h>
#include <log4cxx/logger.h>

#include "communicator.h"
#include "requests.h"

class LunarEnv {
public:
//	inline static const std::string EnvName = "LunarLander-v2";
//	inline static const std::string EnvName = "CartPole-v0";
	const std::string EnvName;

	explicit LunarEnv(std::string url, std::string envName, int instNum = 1);
	~LunarEnv() = default;

	LunarEnv(const LunarEnv&) = delete;
	LunarEnv& operator=(const LunarEnv&) = delete;

	std::pair<cpprl::ActionSpace, cpprl::ActionSpace> init();

	std::vector<float> reset();
	std::vector<float> reset(const int clientId);

	std::tuple<std::vector<float>, float, bool>
		step(const int action, const bool render = false);

	std::tuple<std::vector<float>, std::vector<float>, std::vector<bool>>
		step(const std::vector<long> actions, const bool render = false);

private:
	const int clientNum;
	log4cxx::LoggerPtr logger;
	Communicator comm;
};



#endif /* INC_GYMTEST_LUNARENV_H_ */
