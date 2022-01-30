/*
 * aircnnnet.h
 *
 *  Created on: Apr 5, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_AIRNETS_AIRCNNNET_H_
#define INC_GYMTEST_AIRNETS_AIRCNNNET_H_

#include <torch/torch.h>

struct AirCnnNet: torch::nn::Module {
private:
	torch::nn::Conv2d conv0;
	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Linear fc2;
	torch::nn::Linear fc3;

public:
	AirCnnNet(int outputNum);
	~AirCnnNet() = default;
	AirCnnNet(const AirCnnNet&) = delete;

	torch::Tensor forward(torch::Tensor input);
};



#endif /* INC_GYMTEST_AIRNETS_AIRCNNNET_H_ */
