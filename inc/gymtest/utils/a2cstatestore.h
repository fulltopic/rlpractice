/*
 * a2cstatestore.h
 *
 *  Created on: Apr 30, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_UTILS_A2CSTATESTORE_H_
#define INC_GYMTEST_UTILS_A2CSTATESTORE_H_

#include <vector>
#include <torch/torch.h>

struct A2CState {
	std::vector<float> states;
	std::vector<float> rewards;
	std::vector<bool> dones;
};

struct A2CTensorState {
	torch::Tensor states;
	torch::Tensor actions;
	torch::Tensor rewards;
	torch::Tensor doneMask;
};

#endif /* INC_GYMTEST_UTILS_A2CSTATESTORE_H_ */
