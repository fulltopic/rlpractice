/*
 * rawpolicy.h
 *
 *  Created on: Apr 10, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_TRAIN_RAWPOLICY_H_
#define INC_GYMTEST_TRAIN_RAWPOLICY_H_

#include <torch/torch.h>
#include <vector>
#include <log4cxx/logger.h>

class RawPolicy {
private:
	float epsilon;
	const int actionNum;
	torch::TensorOptions intOpt;
	log4cxx::LoggerPtr logger;

public:
	RawPolicy(float ep, int an);
	~RawPolicy() = default;

	std::vector<int64_t> getActions(torch::Tensor input);
	std::vector<int64_t> getTestActions(torch::Tensor input);

	void updateEpsilon(float e);
	float getEpsilon();
};



#endif /* INC_GYMTEST_TRAIN_RAWPOLICY_H_ */
