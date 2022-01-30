/*
 * airbmnet.h
 *
 *  Created on: Apr 15, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_AIRNETS_AIRCNNBMNET_H_
#define INC_GYMTEST_AIRNETS_AIRCNNBMNET_H_

#include <torch/torch.h>

struct AirCnnBmNet: torch::nn::Module {
private:
	torch::nn::Conv2d conv0;
	torch::nn::BatchNorm2d bm0;
	torch::nn::Conv2d conv1;
	torch::nn::BatchNorm2d bm1;
	torch::nn::Conv2d conv2;
	torch::nn::BatchNorm2d bm2;
	torch::nn::Linear fc2;
	torch::nn::Linear fc3;

public:
	AirCnnBmNet(int outputNum);
	~AirCnnBmNet() = default;
	AirCnnBmNet(const AirCnnBmNet&) = delete;

	torch::Tensor forward(torch::Tensor input);
};




#endif /* INC_GYMTEST_AIRNETS_AIRCNNBMNET_H_ */
