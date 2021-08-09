/*
 * ProbeEnv5.cpp
 *
 *  Created on: Aug 8, 2021
 *      Author: zf
 */




#include "probeenvs/ProbeEnv5.h"
#include <iostream>

ProbeEnv5::ProbeEnv5(int len): inputLen(len), state(len, 0) {}

std::vector<float> ProbeEnv5::reset() {
	updateState();

	return state;
}

std::tuple<std::vector<float>, float, bool>
ProbeEnv5::step(const int action, const bool render) {
	float reward = 0;
	if (action == 1) {
		if (state[0] > 0.5) {
			reward = 1;
		} else {
			reward = -1;
		}
	} else {
		if (state[0] > 0.5) {
			reward = -1;
		} else {
			reward = 1;
		}
	}

	updateState();

	return {state, reward, true};
}

int ProbeEnv5::getInputLen() {
	return inputLen;
}

void ProbeEnv5::updateState() {
	float dice = distribution(generator);
//	std::cout << "dice = " << dice << std::endl;
	float stateValue = 0;
	if (dice == 0) {
		stateValue = -1;
	} else {
		stateValue = 1;
	}

	std::fill(state.begin(), state.end(), stateValue);
}
