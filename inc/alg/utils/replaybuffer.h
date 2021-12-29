/*
 * replaybuffer.hpp
 *
 *  Created on: Dec 29, 2021
 *      Author: zf
 */

#ifndef INC_ALG_UTILS_REPLAYBUFFER_H_
#define INC_ALG_UTILS_REPLAYBUFFER_H_

#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

class ReplayBuffer {
private:
	int curIndex = 0;
	int curSize = 0;
	const int cap;

	const torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	const torch::TensorOptions byteOpt = torch::TensorOptions().dtype(torch::kByte);
//	const torch::TensorOptions charOpt = torch::TensorOptions().dtype(torch::kChar);

	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("rb");

public:
	ReplayBuffer (const int iCap, const at::IntArrayRef& inputShape);
	~ReplayBuffer() = default;
	ReplayBuffer(const ReplayBuffer&) = delete;

	torch::Tensor states;
	torch::Tensor actions;
	torch::Tensor rewards;
	torch::Tensor donesMask;

	void add(torch::Tensor state, torch::Tensor nextState, int action, float reward, float done);
	torch::Tensor getSampleIndex(int batchSize);
};



#endif /* INC_ALG_UTILS_REPLAYBUFFER_H_ */
