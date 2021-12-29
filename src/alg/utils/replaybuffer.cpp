/*
 * replaybuffer.cpp
 *
 *  Created on: Dec 29, 2021
 *      Author: zf
 */



#include "alg/utils/replaybuffer.h"

#include <vector>

ReplayBuffer::ReplayBuffer(const int iCap, const at::IntArrayRef& inputShape): cap(iCap) {
	std::vector<int64_t> stateInputShape;
	stateInputShape.push_back(cap);
	//input state shape = {1, 4, 84, 84};
	for (int i = 1; i < inputShape.size(); i ++) {
		stateInputShape.push_back(inputShape[i]);
	}
	at::IntArrayRef outputShape{ReplayBuffer::cap, 1};

	states = torch::zeros(stateInputShape, byteOpt);
	actions = torch::zeros(outputShape, byteOpt);
	rewards = torch::zeros(outputShape);
	donesMask = torch::zeros(outputShape, byteOpt);

	LOG4CXX_DEBUG(logger, "Replay buffer ready");
}

void ReplayBuffer::add(
		torch::Tensor state, torch::Tensor nextState, int action, float reward, float done) {
		int nextIndex = (curIndex + 1) % cap;

		torch::Tensor inputState = state.to(torch::kByte);
		torch::Tensor inputNextState = nextState.to(torch::kByte);

		states[curIndex].copy_(inputState.squeeze());
		states[nextIndex].copy_(inputNextState.squeeze()); //TODO: Optimize
		actions[curIndex][0] = action;
		rewards[curIndex][0] = reward;
		donesMask[curIndex][0] = done;
		LOG4CXX_DEBUG(logger, "states after copy: " << states[curIndex]);

		curIndex = nextIndex;
		if (curSize < cap) {
			curSize ++;
		}
}

torch::Tensor ReplayBuffer::getSampleIndex(int batchSize) {
	torch::Tensor indices = torch::randint(0, curSize - 1, {batchSize}, longOpt);

	indices = (indices + (curIndex + 1)) % curSize; //No curIndex involved as its next not match

	return indices;
}



