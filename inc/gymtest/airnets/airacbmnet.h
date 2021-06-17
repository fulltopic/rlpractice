/*
 * airacbmnet.h
 *
 *  Created on: Apr 29, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_AIRNETS_AIRACBMNET_H_
#define INC_GYMTEST_AIRNETS_AIRACBMNET_H_

#include <vector>

#include <torch/torch.h>

struct AirACCnnBmNet: torch::nn::Module {
private:
	torch::nn::Conv2d conv0;
	torch::nn::BatchNorm2d bm0;
	torch::nn::Conv2d conv1;
	torch::nn::BatchNorm2d bm1;
	torch::nn::Conv2d conv2;
	torch::nn::BatchNorm2d bm2;
	torch::nn::Linear fc1;
	torch::nn::Linear aOut;
	torch::nn::Linear vOut;

	const int actionNum;
public:
	AirACCnnBmNet(int aNum);
	~AirACCnnBmNet() = default;
	AirACCnnBmNet(const AirACCnnBmNet&) = delete;

	std::vector<torch::Tensor> forward(torch::Tensor input);
};



#endif /* INC_GYMTEST_AIRNETS_AIRACBMNET_H_ */
