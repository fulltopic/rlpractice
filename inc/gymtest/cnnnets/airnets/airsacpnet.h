/*
 * airsacpnet.h
 *
 *  Created on: Sep 21, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_AIRNETS_AIRSACPNET_H_
#define INC_GYMTEST_AIRNETS_AIRSACPNET_H_

#include <torch/torch.h>

struct AirSacPNet: torch::nn::Module {
private:
	torch::nn::Conv2d conv0;
	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Linear fc2;
	torch::nn::Linear fc3;

public:
	AirSacPNet(int outputNum);
	~AirSacPNet() = default;
	AirSacPNet(const AirSacPNet&) = delete;

	torch::Tensor forward(torch::Tensor input);
};





#endif /* INC_GYMTEST_AIRNETS_AIRSACPNET_H_ */
