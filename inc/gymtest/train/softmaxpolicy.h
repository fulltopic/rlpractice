/*
 * softmaxpolicy.h
 *
 *  Created on: Apr 30, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_TRAIN_SOFTMAXPOLICY_H_
#define INC_GYMTEST_TRAIN_SOFTMAXPOLICY_H_


#include <torch/torch.h>
#include <vector>
#include <log4cxx/logger.h>
#include <random>

class SoftmaxPolicy {
private:
	const int actionNum;
	torch::TensorOptions intOpt;
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("softmaxpolicy");

	std::random_device rd;
	std::mt19937 gen;
public:
	SoftmaxPolicy(int an);
	~SoftmaxPolicy() = default;

	std::vector<int64_t> getActions(torch::Tensor input);
	std::vector<int64_t> getTestActions(torch::Tensor input);

};



#endif /* INC_GYMTEST_TRAIN_SOFTMAXPOLICY_H_ */
