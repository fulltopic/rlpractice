/*
 * ProbeEnv2.cpp
 *
 *  Created on: Aug 8, 2021
 *      Author: zf
 */




#include "probeenvs/ProbeEnv2.h"
#include <iostream>

ProbeEnv2::ProbeEnv2(int len): inputLen(len), state(len, 0) {}

//TODO: random -1/1 observation
std::vector<float> ProbeEnv2::reset() {
	updateState();

	return state;
}

std::tuple<std::vector<float>, float, bool>
ProbeEnv2::step(const int action, const bool render) {
	float reward = 1;
	if (state[0] < 0) {
		reward = -1;
	}

	updateState();
	return {state, reward, true};
}

int ProbeEnv2::getInputLen() {
	return inputLen;
}

void ProbeEnv2::updateState() {
	int dice = distribution(generator);
//	std::cout << "dice = " << dice << std::endl;
	float stateValue = 0;
	if (dice == 0) {
		stateValue = -1;
	} else {
		stateValue = 1;
	}

	std::fill(state.begin(), state.end(), stateValue);
}
