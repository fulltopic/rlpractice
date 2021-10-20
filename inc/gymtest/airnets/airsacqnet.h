/*
 * airsacqnet.h
 *
 *  Created on: Sep 21, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_AIRNETS_AIRSACQNET_H_
#define INC_GYMTEST_AIRNETS_AIRSACQNET_H_

#include <torch/torch.h>

struct AirSacQNet: torch::nn::Module {
private:
	torch::nn::Conv2d conv0;
	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Linear fc2;
	torch::nn::Linear fc3;

public:
	AirSacQNet(int outputNum);
	~AirSacQNet() = default;
	AirSacQNet(const AirSacQNet&) = delete;

	torch::Tensor forward(torch::Tensor input);
};





#endif /* INC_GYMTEST_AIRNETS_AIRSACQNET_H_ */
