/*
 * replaybuffer.cpp
 *
 *  Created on: Apr 8, 2021
 *      Author: zf
 */

#include "gymtest/utils/replaybuffer.h"
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <iostream>

ReplayBuffer::ReplayBuffer(int cap): capacity(cap),
	states(capacity, std::vector<float>()),
	nextStates(capacity, std::vector<float>()),
	rewards(capacity, std::vector<float>()),
	dones(capacity, std::vector<bool>()),
	actions(capacity, std::vector<int64_t>()),
	index(0),
	distribution(0, capacity - 1),
	indices(cap, 0),
	indiceIndex(0)
{
	for (int i = 0; i < cap; i ++) {
		indices[i] = i;
	}

	std::srand(unsigned (std::time(0)));
}

void ReplayBuffer::put(std::vector<float> state, std::vector<float> nextState,
		std::vector<float> reward, std::vector<long> action, std::vector<bool> done) {
//	std::cout << "action size = " << action.size() << std::endl;
	states[index].resize(state.size());
	std::copy(state.begin(), state.end(), states[index].begin());
	nextStates[index].resize(nextState.size());
	std::copy(nextState.begin(), nextState.end(), nextStates[index].begin());
	rewards[index].resize(reward.size());
	std::copy(reward.begin(), reward.end(), rewards[index].begin());
//	std::cout << "restore reward size = " << rewards[index].size() << std::endl;
	actions[index].resize(action.size());
	std::copy(action.begin(), action.end(), actions[index].begin());
//	std::cout << "restore action size = " << actions[index].size() << std::endl;
	dones[index].resize(done.size());
	std::copy(done.begin(), done.end(), dones[index].begin());

//
//	states[index] = state;
//	nextStates[index] = nextState;
//	rewards[index] = reward;
//	actions[index] = action;
//	dones[index] = done;

	index = (index + 1) % capacity;
	count = std::min(count + 1, capacity);
}

int ReplayBuffer::randSelect() {
//	return distribution(generator);
	if (indiceIndex == 0) {
		std::random_shuffle(indices.begin(), indices.end());
	}
	int randIndex = indices[indiceIndex];
	indiceIndex = (indiceIndex + 1) % capacity;

//	std::cout << "get randam index " << indiceIndex << std::endl;
	return randIndex % count;
}

int ReplayBuffer::getIndex() {
	return index;
}



