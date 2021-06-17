/*
 * airacnet.h
 *
 *  Created on: May 4, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_AIRNETS_AIRACNET_H_
#define INC_GYMTEST_AIRNETS_AIRACNET_H_



#include <vector>

#include <torch/torch.h>

struct AirACCnnNet: torch::nn::Module {
private:
	torch::nn::Conv2d conv0;
	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Linear fc1;
	torch::nn::Linear aOut;
	torch::nn::Linear vOut;

	const int actionNum;
public:
	AirACCnnNet(int aNum);
	~AirACCnnNet() = default;
	AirACCnnNet(const AirACCnnNet&) = delete;

	std::vector<torch::Tensor> forward(torch::Tensor input);
};



#endif /* INC_GYMTEST_AIRNETS_AIRACNET_H_ */
