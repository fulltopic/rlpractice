/*
 * airdueling.h
 *
 *  Created on: Apr 20, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_AIRNETS_AIRDUELING_H_
#define INC_GYMTEST_AIRNETS_AIRDUELING_H_


#include <torch/torch.h>

struct AirCnnDuelNet: torch::nn::Module {
private:
	torch::nn::Conv2d conv0;
	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Linear vfc0;
	torch::nn::Linear vfc1;
	torch::nn::Linear afc0;
	torch::nn::Linear afc1;

	const int actionNum;

public:
	AirCnnDuelNet(int outputNum);
	~AirCnnDuelNet() = default;
	AirCnnDuelNet(const AirCnnDuelNet&) = delete;

	torch::Tensor forward(torch::Tensor input);
};



#endif /* INC_GYMTEST_AIRNETS_AIRDUELING_H_ */
