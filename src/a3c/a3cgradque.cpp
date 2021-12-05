/*
 * a3cgradque.cpp
 *
 *  Created on: Dec 2, 2021
 *      Author: zf
 */


#include "a3c/a3cgradque.h"

A3CGradQueue::A3CGradQueue(std::size_t maxThreshold, uint64_t iLogGap, std::string logPath):
	maxLimit(maxThreshold),
	logGap(iLogGap),
	tLogger(logPath.c_str())
{
	assert(maxLimit >= 2);//avoid empty wait
}


void A3CGradQueue::push(std::vector<torch::Tensor>& ts) {
	std::unique_lock<std::mutex> lock(qMutex);

	q.push(ts);

	pushNum ++;
	if ((pushNum % logGap) == 0) {
		tLogger.add_scalar("server/queue", pushNum, (float)q.size());
	}

	if (q.size() > maxLimit) {
		LOG4CXX_ERROR(logger, "Grad beyond server capability, to clear");

		while (q.size() > (maxLimit / 2)) {
			q.pop();
		}
	}

	qCond.notify_all();
}

std::vector<torch::Tensor> A3CGradQueue::pop() {
	std::unique_lock<std::mutex> lock(qMutex);

	while (q.empty()) {
		qCond.wait(lock);
	}

	auto data = q.front();
	q.pop();

	return data;
}
