/*
 * airsacqb1net.h
 *
 *  Created on: Nov 4, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_AIRNETS_AIRSACQB1NET_H_
#define INC_GYMTEST_AIRNETS_AIRSACQB1NET_H_


#include <torch/torch.h>

struct AirSacQB1Net: torch::nn::Module {
private:
	torch::nn::Conv2d conv0;
	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Linear fc2;
	torch::nn::Linear fc3;

public:
	AirSacQB1Net(int outputNum);
	~AirSacQB1Net() = default;
	AirSacQB1Net(const AirSacQB1Net&) = delete;

	torch::Tensor forward(torch::Tensor input);
};




#endif /* INC_GYMTEST_AIRNETS_AIRSACQB1NET_H_ */
