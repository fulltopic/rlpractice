/*
 * ProbeEnvWrapper.cpp
 *
 *  Created on: Aug 8, 2021
 *      Author: zf
 */

#include "probeenvs/ProbeEnvWrapper.h"

ProbeEnvWrapper::ProbeEnvWrapper(int inputLen, int envId, int num) {
	impl = ProbeVecEnv::CreateEnv(inputLen, envId, num);
}


std::tuple<std::vector<float>, std::vector<float>, std::vector<bool>>
ProbeEnvWrapper::step(const std::vector<long> actions, const bool render) {
	return impl->step(actions, render);
}

std::vector<float> ProbeEnvWrapper::reset() {
	return impl->reset();
}

int ProbeEnvWrapper::getInputLen() {
	return impl->getInputLen();
}
