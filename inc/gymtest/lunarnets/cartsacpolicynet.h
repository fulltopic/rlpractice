/*
 * cartsacpolicynet.h
 *
 *  Created on: Sep 16, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_LUNARNETS_CARTSACPOLICYNET_H_
#define INC_GYMTEST_LUNARNETS_CARTSACPOLICYNET_H_


#include <torch/torch.h>

#include "netconfig.h"

struct CartSacPolicyNet: torch::nn::Module {
private:
	torch::nn::Linear fc0;
	torch::nn::Linear fcOutput;

	const int inputNum;
	const int outputNum;
public:
	CartSacPolicyNet(int intput, int output);
	~CartSacPolicyNet() = default;
	CartSacPolicyNet(const CartSacPolicyNet&) = delete;

	torch::Tensor forward(torch::Tensor input);
};




#endif /* INC_GYMTEST_LUNARNETS_CARTSACPOLICYNET_H_ */
