/*
 * CartSacQNet.h
 *
 *  Created on: Sep 16, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_LUNARNETS_CARTSACQNET_H_
#define INC_GYMTEST_LUNARNETS_CARTSACQNET_H_

#include <torch/torch.h>

#include "netconfig.h"

struct CartSacQNet: torch::nn::Module {
private:
	torch::nn::Linear fc0;
	torch::nn::Linear fcOutput;

	const int inputNum;
	const int outputNum;
public:
	CartSacQNet(int intput, int output);
	~CartSacQNet() = default;
	CartSacQNet(const CartSacQNet&) = delete;

	torch::Tensor forward(torch::Tensor input);
};







#endif /* INC_GYMTEST_LUNARNETS_CARTSACQNET_H_ */
