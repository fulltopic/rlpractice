/*
 * appodataq.cpp
 *
 *  Created on: Dec 15, 2021
 *      Author: zf
 */

#include "alg/appo/appodataq.h"

#include <torch/torch.h>

AsyncPPODataQ::AsyncPPODataQ(std::size_t maxThreshold):
	maxLimit(maxThreshold)
{
	assert(maxLimit >= 2);//avoid empty wait
}


void AsyncPPODataQ::push(std::vector<torch::Tensor>&& ts) {
	std::unique_lock<std::mutex> lock(qMutex);

	while (q.size() >= maxLimit) {
		pushCond.wait(lock);
	}

	q.push(ts);

	pushNum ++;
	itemNum += ts[0].sizes()[0];

//	if (q.size() > maxLimit) {
//		LOG4CXX_ERROR(logger, "Grad beyond server capability, to clear");
//
//		while (q.size() > maxLimit) {
//			auto data = q.front();
//			q.pop();
//
//			itemNum -= data[0].sizes()[0];
//		}
//	}

	popCond.notify_all();
}

std::vector<torch::Tensor> AsyncPPODataQ::pop(const int itemLen) {
	std::unique_lock<std::mutex> lock(qMutex);

	//TODO: log q traffic
	while (itemNum < itemLen) {
		popCond.wait(lock);
	}

	int returnItemLen = 0;

	std::vector<torch::Tensor> states;
	std::vector<torch::Tensor> returns;
	std::vector<torch::Tensor> gaes;
	std::vector<torch::Tensor> oldPis;
	std::vector<torch::Tensor> actions;

	while (returnItemLen < itemLen)	{
		auto data = q.front();
		q.pop();

		returnItemLen += data[0].sizes()[0];
		itemNum -= data[0].sizes()[0];

		states.push_back(data[0]);
		returns.push_back(data[1]);
		gaes.push_back(data[2]);
		oldPis.push_back(data[3]);
		actions.push_back(data[4]);
	}

	pushCond.notify_all();

	auto stateTensor = torch::cat(states, 0);
	auto returnTensor = torch::cat(returns, 0);
	auto gaeTensor = torch::cat(gaes, 0);
	auto oldPiTensor = torch::cat(oldPis, 0);
	auto actionTensor = torch::cat(actions, 0);

	return {stateTensor, returnTensor, gaeTensor, oldPiTensor, actionTensor};
}



