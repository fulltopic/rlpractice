/*
 * appodataq.h
 *
 *  Created on: Dec 15, 2021
 *      Author: zf
 */

#ifndef INC_ALG_APPO_APPODATAQ_H_
#define INC_ALG_APPO_APPODATAQ_H_

#include <queue>
#include <thread>
#include <mutex>
#include <vector>
#include <string>

#include <torch/torch.h>

#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

class AsyncPPODataQ {
private:
	std::mutex qMutex;
	std::condition_variable pushCond;
	std::condition_variable popCond;

	std::queue<std::vector<torch::Tensor> > q;

	const std::size_t maxLimit = 20;
	uint64_t pushNum = 0;
	uint64_t itemNum = 0;

	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("appodataq");

	AsyncPPODataQ(const AsyncPPODataQ&) = delete;

public:
	AsyncPPODataQ(std::size_t maxThreshold);
	~AsyncPPODataQ() = default;

	void push(std::vector<torch::Tensor>&& ts);
	std::vector<torch::Tensor> pop(const int itemLen);

	inline auto getStoreSize() const { return q.size(); }
};





#endif /* INC_ALG_APPO_APPODATAQ_H_ */
