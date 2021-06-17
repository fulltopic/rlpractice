/*
 * cartacnet.h
 *
 *  Created on: Apr 30, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_LUNARNETS_CARTACNET_H_
#define INC_GYMTEST_LUNARNETS_CARTACNET_H_

#include <torch/torch.h>
#include <vector>
#include "netconfig.h"

struct CartACFcNet: torch::nn::Module {
private:
	torch::nn::Linear fc0;
	torch::nn::Linear vOutput;
	torch::nn::Linear aOutput;

	const int inputNum;
	const int outputNum;
public:
	CartACFcNet(int intput, int output);
	~CartACFcNet() = default;

	std::vector<torch::Tensor> forward(torch::Tensor input);
//	torch::Tensor getLoss(torch::Tensor input, torch::Tensor returns);
};





#endif /* INC_GYMTEST_LUNARNETS_CARTACNET_H_ */
