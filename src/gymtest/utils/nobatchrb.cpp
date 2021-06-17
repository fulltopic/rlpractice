/*
 * nobatchrb.cpp
 *
 *  Created on: Apr 11, 2021
 *      Author: zf
 */


#include "gymtest/utils/nobatchrb.h"
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <iostream>

NoBatchRB::NoBatchRB(int cap): capacity(cap),
	states(capacity, std::vector<float>()),
	nextStates(capacity, std::vector<float>()),
	rewards(capacity, 0),
	dones(capacity, false),
	actions(capacity, 0),
	indices(cap, 0)
{
	for (int i = 0; i < cap; i ++) {
		indices[i] = i;
	}

	std::srand(unsigned (std::time(0)));
}

void NoBatchRB::put(std::vector<float> state, std::vector<float> nextState,
	std::vector<float> reward, std::vector<long> action, std::vector<bool> done, const int batchSize) {
	const auto stateSize = state.size() / batchSize;
//	const auto rewardSize = reward.size() / batchSize;
//	const auto actionSize = action.size() / batchSize;
//	const auto doneSize = done.size() / batchSize;

	for (int i = 0; i < batchSize; i ++) {
		states[index].resize(stateSize);
		std::copy(state.begin() + i * stateSize, state.begin() + (i + 1) * stateSize, states[index].begin());
		nextStates[index].resize(stateSize);
		std::copy(nextState.begin() + i * stateSize, nextState.begin() + (i + 1) * stateSize, nextStates[index].begin());

		rewards[index] = reward[i];
		actions[index] = action[i];
		dones[index] = done[i];

		index = (index + 1) % capacity;
	}

	count = std::min(count + batchSize, capacity);
}

int NoBatchRB::randSelect() {
//	return distribution(generator);
	if (indiceIndex == 0) {
		std::random_shuffle(indices.begin(), indices.end());
	}
	int randIndex = indices[indiceIndex] % count;
	indiceIndex = (indiceIndex + 1) % capacity;

//	std::cout << "get randam index " << indiceIndex << std::endl;
	return randIndex;
}




