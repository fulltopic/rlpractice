/*
 * cartqnet.h
 *
 *  Created on: Aug 23, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_LUNARNETS_CARTQNET_H_
#define INC_GYMTEST_LUNARNETS_CARTQNET_H_

#include <torch/torch.h>

#include "netconfig.h"

struct CartFcQNet: torch::nn::Module {
private:
	torch::nn::Linear fc0;
	torch::nn::Linear fcOutput;

	const int inputNum;
	const int outputNum;
public:
	CartFcQNet(int intput, int output);
	~CartFcQNet() = default;

	torch::Tensor forward(torch::Tensor input);
//	torch::Tensor getLoss(torch::Tensor input, torch::Tensor returns);
};







#endif /* INC_GYMTEST_LUNARNETS_CARTQNET_H_ */
