/*
 * ProbeEnvWrapper.h
 *
 *  Created on: Aug 8, 2021
 *      Author: zf
 */

#ifndef INC_PROBEENVS_PROBEENVWRAPPER_H_
#define INC_PROBEENVS_PROBEENVWRAPPER_H_

#include "probevecenv.h"
#include <memory>

class ProbeEnvWrapper {
public:
	ProbeEnvWrapper(int inputLen, int envId, int num);
	~ProbeEnvWrapper() = default;

	ProbeEnvWrapper(const ProbeEnvWrapper& o) = delete;


	std::tuple<std::vector<float>, std::vector<float>, std::vector<bool>>
		step(const std::vector<long> actions, const bool render = false);

	std::vector<float> reset();

	int getInputLen();
protected:
	std::unique_ptr<ProbeVecEnv> impl;
};



#endif /* INC_PROBEENVS_PROBEENVWRAPPER_H_ */
