/*
 * probeenv1.cpp
 *
 *  Created on: Aug 7, 2021
 *      Author: zf
 */

#include "probeenvs/ProbeEnv1.h"

ProbeEnv1::ProbeEnv1(int len): inputLen(len) {}


std::vector<float> ProbeEnv1::reset() {
	return std::vector<float>(inputLen, 0);
}

std::tuple<std::vector<float>, float, bool>
ProbeEnv1::step(const int action, const bool render) {
	return {std::vector<float>(inputLen, 0), 1, true};
}

int ProbeEnv1::getInputLen() {
	return inputLen;
}
