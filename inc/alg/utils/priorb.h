/*
 * priorb.h
 *
 *  Created on: Dec 30, 2021
 *      Author: zf
 */

#ifndef INC_ALG_UTILS_PRIORB_H_
#define INC_ALG_UTILS_PRIORB_H_

#include <torch/torch.h>

#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <algorithm>
#include <cstdlib>

#include "stree.h"
#include "mheap.hpp"

class PrioRbFloatMaxPredFunc {
public:
	bool operator()(const float a, const float b);
};

class PrioReplayBuffer {
private:
	int curIndex = 0;
	int curSize = 0;
	const int cap;

	const float epsilon;

	PrioRbFloatMaxPredFunc maxPrioFunc;
	MHeap<float, PrioRbFloatMaxPredFunc> maxPrioHeap;
	SegTree segTree;

	const torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	const torch::TensorOptions byteOpt = torch::TensorOptions().dtype(torch::kByte);
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("dqn");

public:
	PrioReplayBuffer (const int iCap, const at::IntArrayRef& inputShape, float iEpsilon);
	~PrioReplayBuffer() = default;
	PrioReplayBuffer(const PrioReplayBuffer&) = delete;

	torch::Tensor states;
	torch::Tensor actions;
	torch::Tensor rewards;
	torch::Tensor donesMask;

	//Store states and rewards after normalization
	void add(torch::Tensor state, torch::Tensor nextState, int action, float reward, float done);
	void update(torch::Tensor indiceTensor, torch::Tensor prioTensor);
	std::pair<torch::Tensor, torch::Tensor> getSampleIndex(int batchSize);
	inline int size() { return curSize;}
	inline float sum() { return segTree.getSum();}

	void print();
};



#endif /* INC_ALG_UTILS_PRIORB_H_ */
