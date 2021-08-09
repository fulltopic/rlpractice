/*
 * probevecenv.h
 *
 *  Created on: Aug 7, 2021
 *      Author: zf
 */

#ifndef INC_PROBEENVS_PROBEVECENV_H_
#define INC_PROBEENVS_PROBEVECENV_H_

#include <vector>
#include <memory>

#include "ProbeEnv.h"

class ProbeVecEnv {
public:
	~ProbeVecEnv() = default;

	static std::unique_ptr<ProbeVecEnv> CreateEnv(const int inputLen, const int envId, const int num);
	ProbeVecEnv() = default;

	std::tuple<std::vector<float>, std::vector<float>, std::vector<bool>>
		step(const std::vector<long> actions, const bool render = false);

	std::vector<float> reset();

	int getInputLen();

protected:
	std::vector<std::unique_ptr<ProbeEnv>> envs;
};



#endif /* INC_PROBEENVS_PROBEVECENV_H_ */
