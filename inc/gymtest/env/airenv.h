/*
 * airenv.h
 *
 *  Created on: Apr 5, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_ENV_AIRENV_H_
#define INC_GYMTEST_ENV_AIRENV_H_

#include <string>
#include <vector>

#include <cpprl/cpprl.h>
#include <log4cxx/logger.h>

#include "communicator.h"
#include "requests.h"

class AirEnv {
public:
//	inline static const std::string EnvName = "LunarLander-v2";
//	inline static const std::string EnvName = "CartPole-v0";
	const std::string EnvName;

	explicit AirEnv(std::string url, std::string envName = "Alien-v0", int instNum = 1);
	~AirEnv() = default;

	AirEnv(const AirEnv&) = delete;
	AirEnv& operator=(const AirEnv&) = delete;

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




#endif /* INC_GYMTEST_ENV_AIRENV_H_ */
