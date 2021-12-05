/*
 * a3cgradque.h
 *
 *  Created on: Dec 2, 2021
 *      Author: zf
 */

#ifndef INC_A3C_A3CGRADQUE_H_
#define INC_A3C_A3CGRADQUE_H_

#include <queue>
#include <thread>
#include <mutex>
#include <vector>
#include <string>

#include <torch/torch.h>

#include <tensorboard_logger.h>

#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

class A3CGradQueue {
private:
	std::mutex qMutex;
	std::condition_variable qCond;

	std::queue<std::vector<torch::Tensor> > q;

	const std::size_t maxLimit = 20;
	const uint64_t logGap = 10;
	uint64_t pushNum = 0;

	TensorBoardLogger tLogger;
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("a3cgradque");

	A3CGradQueue(const A3CGradQueue&) = delete;

public:
	A3CGradQueue(std::size_t maxThreshold, uint64_t iLogGap, std::string logPath);
	~A3CGradQueue() = default;

	void push(std::vector<torch::Tensor>& ts);
	std::vector<torch::Tensor> pop();
};


#endif /* INC_A3C_A3CGRADQUE_H_ */
