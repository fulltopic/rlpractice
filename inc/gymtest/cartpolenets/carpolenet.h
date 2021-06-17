/*
 * carpolenet.h
 *
 *  Created on: Apr 22, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_CARPOLENET_H_
#define INC_GYMTEST_CARPOLENET_H_

#include <torch/torch.h>

struct CartCnnBmNet: torch::nn::Module {
private:
	torch::nn::Conv2d conv0;
	torch::nn::BatchNorm2d bm0;
	torch::nn::Conv2d conv1;
	torch::nn::BatchNorm2d bm1;
	torch::nn::Conv2d conv2;
	torch::nn::BatchNorm2d bm2;
	torch::nn::Linear fc;

public:
	CartCnnBmNet(int outputNum);
	~CartCnnBmNet() = default;
	CartCnnBmNet(const CartCnnBmNet&) = delete;

	torch::Tensor forward(torch::Tensor input);
};




#endif /* INC_GYMTEST_CARPOLENET_H_ */
