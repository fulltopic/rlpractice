/*
 * ProbeEnv3.cpp
 *
 *  Created on: Aug 8, 2021
 *      Author: zf
 */



#include "probeenvs/ProbeEnv3.h"
#include <iostream>

ProbeEnv3::ProbeEnv3(int len): inputLen(len), state(len, 0) {}

//TODO: random -1/1 observation
std::vector<float> ProbeEnv3::reset() {
	return state;
}

std::tuple<std::vector<float>, float, bool>
ProbeEnv3::step(const int action, const bool render) {
	updateState();

	float reward = 0;
	bool done = false;
	if (state[0] > 0) {
		reward = 1;
		done = true;
	}

	return {state, reward, done};
}

int ProbeEnv3::getInputLen() {
	return inputLen;
}

void ProbeEnv3::updateState() {
	float stateValue = 1 - state[0];

	std::fill(state.begin(), state.end(), stateValue);
}
