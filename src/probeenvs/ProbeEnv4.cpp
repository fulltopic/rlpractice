/*
 * ProbeEnv4.cpp
 *
 *  Created on: Aug 8, 2021
 *      Author: zf
 */




#include "probeenvs/ProbeEnv4.h"
#include <iostream>

ProbeEnv4::ProbeEnv4(int len): inputLen(len) {}

//TODO: random -1/1 observation
std::vector<float> ProbeEnv4::reset() {
	return std::vector<float>(inputLen, 0);
}

std::tuple<std::vector<float>, float, bool>
ProbeEnv4::step(const int action, const bool render) {
	float reward = -1;
	if (action == 1) {
		reward = 1;
	}

	return {std::vector<float>(inputLen, 0), reward, true};
}

int ProbeEnv4::getInputLen() {
	return inputLen;
}


