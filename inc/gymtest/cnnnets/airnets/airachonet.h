/*
 * airachonet.h
 *
 *  Created on: Jun 4, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_AIRNETS_AIRACHONET_H_
#define INC_GYMTEST_AIRNETS_AIRACHONET_H_


#include <vector>

#include <torch/torch.h>

struct AirACHONet: torch::nn::Module {
private:
	torch::nn::Conv2d conv0;
	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Linear afc;
	torch::nn::Linear vfc;
	torch::nn::Linear aOut;
	torch::nn::Linear vOut;

	const int actionNum;
public:
	AirACHONet(int aNum);
	~AirACHONet() = default;
	AirACHONet(const AirACHONet&) = delete;

	std::vector<torch::Tensor> forward(torch::Tensor input);
};



#endif /* INC_GYMTEST_AIRNETS_AIRACHONET_H_ */
