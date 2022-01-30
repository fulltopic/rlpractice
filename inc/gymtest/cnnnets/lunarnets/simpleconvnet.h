/*
 * simpleconvnet.h
 *
 *  Created on: Apr 5, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_LUNARNETS_SIMPLECONVNET_H_
#define INC_GYMTEST_LUNARNETS_SIMPLECONVNET_H_

#include <torch/torch.h>

//#include "netconfig.h"

struct SimpleConvNet: torch::nn::Module {
private:
	torch::nn::Conv2d fc0;

	torch::nn::Linear fcOutput;

public:
	SimpleConvNet();
	~SimpleConvNet() = default;

	torch::Tensor forward(torch::Tensor input);
//	torch::Tensor getLoss(torch::Tensor input, torch::Tensor returns);
};




#endif /* INC_GYMTEST_LUNARNETS_SIMPLECONVNET_H_ */
