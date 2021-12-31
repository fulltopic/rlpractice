/*
 * priorb.cpp
 *
 *  Created on: Dec 30, 2021
 *      Author: zf
 */


#include "alg/utils/priorb.h"

bool PrioRbFloatMaxPredFunc::operator()(const float a, const float b) {
	return a > b;
}

PrioReplayBuffer::PrioReplayBuffer(const int iCap, const at::IntArrayRef& inputShape, float iEps)
	: cap(iCap),
	  epsilon(iEps),
	  maxPrioHeap(iCap, maxPrioFunc),
	  segTree(iCap)
{
	std::vector<int64_t> stateInputShape;
	stateInputShape.push_back(cap);
	//input state shape = {1, 4, 84, 84};
	for (int i = 1; i < inputShape.size(); i ++) {
		stateInputShape.push_back(inputShape[i]);
	}
	at::IntArrayRef outputShape{PrioReplayBuffer::cap, 1};

//	LOG4CXX_INFO(logger, "stateInputShape " << stateInputShape);
	//atari
	states = torch::zeros(stateInputShape, byteOpt);
	//non-atari
//	states = torch::zeros(stateInputShape);

	actions = torch::zeros(outputShape, byteOpt);
	rewards = torch::zeros(outputShape);
	donesMask = torch::zeros(outputShape, byteOpt);

	LOG4CXX_DEBUG(logger, "Replay buffer ready");
}


void PrioReplayBuffer::add(torch::Tensor state, torch::Tensor nextState, int action, float reward, float done) {

	//atari
	torch::Tensor inputState = state.to(torch::kByte);
	torch::Tensor inputNextState = nextState.to(torch::kByte);
	//non atari
//	torch::Tensor inputState = state;
//	torch::Tensor inputNextState = nextState;

	{
		//For log
		bool isSame = states[curIndex].equal(inputState);
		LOG4CXX_DEBUG(logger, curIndex << ": The state same? " << isSame);
		if (!isSame) {
			LOG4CXX_DEBUG(logger, "curState: " << states[curIndex]);
			LOG4CXX_DEBUG(logger, "inputState: " << inputState);
			LOG4CXX_DEBUG(logger, "nextState: " << inputNextState);
		}
	}

	int nextIndex = (curIndex + 1) % cap;

	LOG4CXX_DEBUG(logger, "current shape " << states[curIndex].sizes());
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

	float maxPrio = std::max(maxPrioHeap.getM(), epsilon);
	LOG4CXX_DEBUG(logger, "add " << maxPrio);
	segTree.add(maxPrio);
	maxPrioHeap.add(maxPrio);
}

std::pair<torch::Tensor, torch::Tensor> PrioReplayBuffer::getSampleIndex(int batchSize) {
//	torch::Tensor indices = torch::randint(0, curSize, {batchSize}, longOpt);
//
//	return indices;
	auto rc = segTree.sample(batchSize);
	std::vector<int> indice = std::get<0>(rc);
	std::vector<float> prios = std::get<1>(rc);

	torch::Tensor indiceTensor = torch::zeros({batchSize}, longOpt);
	torch::Tensor prioTensor = torch::zeros({batchSize});

	for (int i = 0; i < batchSize; i ++) {
		indiceTensor[i] = indice[i];
		prioTensor[i] = prios[i];
	}

	LOG4CXX_DEBUG(logger, "sample indice: " << indiceTensor);
	LOG4CXX_DEBUG(logger, "sample prio: " << prioTensor);
	return {indiceTensor, prioTensor};
}

void PrioReplayBuffer::update(torch::Tensor indiceTensor, torch::Tensor prioTensor) {
	int batchSize = indiceTensor.numel();
	LOG4CXX_DEBUG(logger, "To update " << batchSize);
	LOG4CXX_DEBUG(logger, "To update: " << indiceTensor);
	LOG4CXX_DEBUG(logger, "before convert: " << prioTensor);
	int64_t* indice = indiceTensor.to(torch::kCPU).data_ptr<int64_t>();
	prioTensor = prioTensor.to(torch::kCPU);
	std::vector<float> prios(prioTensor.data_ptr<float>(), prioTensor.data_ptr<float>() + batchSize);

	for (int i = 0; i < batchSize; i ++) {
		maxPrioHeap.update(indice[i], prios[i]);
		segTree.update(indice[i], prios[i]);

		LOG4CXX_DEBUG(logger, "update " << indice[i] << " into " << prios[i]);
	}
}

void PrioReplayBuffer::print() {
	std::cout << "stree \n" << segTree << std::endl;
	std::cout << "heap \n" << maxPrioHeap << std::endl;
}
