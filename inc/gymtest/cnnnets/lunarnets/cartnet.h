/*
 * cartnet.h
 *
 *  Created on: Apr 24, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_LUNARNETS_CARTNET_H_
#define INC_GYMTEST_LUNARNETS_CARTNET_H_

#include <torch/torch.h>

//#include "netconfig.h"

struct CartFcNet: torch::nn::Module {
private:
	torch::nn::Linear fc0;
	torch::nn::Linear fcOutput;

	const int inputNum;
	const int outputNum;
public:
	CartFcNet(int intput, int output);
	~CartFcNet() = default;

	torch::Tensor forward(torch::Tensor input);
//	torch::Tensor getLoss(torch::Tensor input, torch::Tensor returns);
};





#endif /* INC_GYMTEST_LUNARNETS_CARTNET_H_ */
