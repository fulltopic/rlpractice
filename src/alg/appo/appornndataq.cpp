/*
 * appornndataq.cpp
 *
 *  Created on: Mar 6, 2022
 *      Author: zf
 */



#include "alg/rnn/appo/rnnappodataq.h"

#include <torch/torch.h>

AsyncRnnPPODataQ::AsyncRnnPPODataQ(std::size_t maxThreshold):
	maxLimit(maxThreshold)
{
	assert(maxLimit >= 2);//avoid empty wait
}


void AsyncRnnPPODataQ::push(std::vector<torch::Tensor>&& ts) {
	std::unique_lock<std::mutex> lock(qMutex);

	while (q.size() >= maxLimit) {
		pushCond.wait(lock);
	}

	q.push(ts);


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

std::vector<torch::Tensor> AsyncRnnPPODataQ::pop() {
	std::unique_lock<std::mutex> lock(qMutex);

	//TODO: log q traffic
	while (q.empty()) {
		popCond.wait(lock);
	}

	auto data = q.front();
	q.pop();

	pushCond.notify_all();

	return data;
}




