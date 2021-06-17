/*
 * fcnet.h
 *
 *  Created on: Apr 4, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_LUNARNETS_LUNARFCNET_H_
#define INC_GYMTEST_LUNARNETS_LUNARFCNET_H_

#include <torch/torch.h>

#include "netconfig.h"

struct LunarFcNet: torch::nn::Module {
private:
	torch::nn::Linear fc0;
	torch::nn::Linear fcOutput;

public:
	LunarFcNet();
	~LunarFcNet() = default;

	torch::Tensor forward(torch::Tensor input);
//	torch::Tensor getLoss(torch::Tensor input, torch::Tensor returns);
};



#endif /* INC_GYMTEST_LUNARNETS_LUNARFCNET_H_ */
