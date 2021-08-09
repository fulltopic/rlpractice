/*
 * probevecenv.cpp
 *
 *  Created on: Aug 7, 2021
 *      Author: zf
 */


#include "probeenvs/probevecenv.h"
#include "probeenvs/ProbeEnv1.h"
#include "probeenvs/ProbeEnv2.h"
#include "probeenvs/ProbeEnv3.h"
#include "probeenvs/ProbeEnv4.h"
#include "probeenvs/ProbeEnv5.h"
#include "probeenvs/ProbeEnv.h"

#include "gymtest/env/envutils.h"

std::unique_ptr<ProbeVecEnv> ProbeVecEnv::CreateEnv(const int inputLen, const int envId, const int num) {
	auto env = std::make_unique<ProbeVecEnv>();

	for (int i = 0; i < num; i ++) {
		switch (envId) {
		case 1:
			env->envs.push_back(std::make_unique<ProbeEnv1>(inputLen));
			break;
		case 2:
			env->envs.push_back(std::make_unique<ProbeEnv2>(inputLen));
			break;
		case 3:
			env->envs.push_back(std::make_unique<ProbeEnv3>(inputLen));
			break;
		case 4:
			env->envs.push_back(std::make_unique<ProbeEnv4>(inputLen));
			break;
		case 5:
			env->envs.push_back(std::make_unique<ProbeEnv5>(inputLen));
			break;
		}
	}

	return env;
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<bool>>
	ProbeVecEnv::step(const std::vector<long> actions, const bool render) {
	//TODO: actions size match envs size

	std::vector<std::vector<float>> nextStates;
	std::vector<float> rewards;
	std::vector<bool> dones;

	for (int i = 0; i < envs.size(); i ++) {
		auto rc = envs[i]->step(actions[i], render);
		nextStates.push_back(std::get<0>(rc));
		rewards.push_back(std::get<1>(rc));
		dones.push_back(std::get<2>(rc));
	}

	auto nextStateFlat = EnvUtils::FlattenVector(nextStates);

	return {nextStateFlat, rewards, dones};
}

//TODO: random some step in beginning
std::vector<float> ProbeVecEnv::reset() {
	std::vector<std::vector<float>> states;

	for (int i = 0; i < envs.size(); i ++) {
		auto rc = envs[i]->reset();
		states.push_back(rc);
	}

	auto stateFlat = EnvUtils::FlattenVector(states);
	return stateFlat;
}

int ProbeVecEnv::getInputLen() {
	if (envs.size() == 0) {
		return 0;
	} else {
		return envs[0]->getInputLen();
	}
}
