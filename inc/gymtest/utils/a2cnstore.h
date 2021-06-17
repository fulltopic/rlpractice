/*
 * a2cnstore.h
 *
 *  Created on: May 5, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_UTILS_A2CNSTORE_H_
#define INC_GYMTEST_UTILS_A2CNSTORE_H_



#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <vector>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/a2cstatestore.h"
#include "gymtest/utils/stats.h"
#include "gymtest/utils/lossstats.h"

class A2CNStorage {
private:
	const torch::Device deviceType;


	std::vector<torch::Tensor> values;
	std::vector<torch::Tensor> advLogs;
	std::vector<torch::Tensor> rewards;
	std::vector<torch::Tensor> doneMasks;

	torch::Tensor entropy;

	const int cap;
	uint32_t updateSeq = 0;

	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("a2cnstore");

public:
	A2CNStorage(const A2CNStorage&) = delete;
	~A2CNStorage() = default;

	A2CNStorage(torch::Device devType, int size);

	void reset();
	void put(torch::Tensor value, torch::Tensor actLinearOutput, torch::Tensor action, torch::Tensor reward, torch::Tensor doneMask);
	torch::Tensor getLoss(torch::Tensor finalValue, float gamma, float entropyFactor, float valueFactor, Stats& stat, LossStats& lossStat);

	inline bool toUpdate() { return values.size() == cap;}
};



#endif /* INC_GYMTEST_UTILS_A2CNSTORE_H_ */
