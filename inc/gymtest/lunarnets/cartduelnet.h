/*
 * cartduelnet.h
 *
 *  Created on: Apr 25, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_LUNARNETS_CARTDUELNET_H_
#define INC_GYMTEST_LUNARNETS_CARTDUELNET_H_

#include <torch/torch.h>

#include "netconfig.h"

struct CartDuelFcNet: torch::nn::Module {
private:
	torch::nn::Linear fc0;
	torch::nn::Linear vOutput;
	torch::nn::Linear aOutput;

	const int inputNum;
	const int outputNum;
public:
	CartDuelFcNet(int intput, int output);
	~CartDuelFcNet() = default;

	torch::Tensor forward(torch::Tensor input);
//	torch::Tensor getLoss(torch::Tensor input, torch::Tensor returns);
};





#endif /* INC_GYMTEST_LUNARNETS_CARTDUELNET_H_ */
