/*
 * airacbmsmallkernelnet.h
 *
 *  Created on: May 7, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_AIRNETS_AIRACBMSMALLKERNELNET_H_
#define INC_GYMTEST_AIRNETS_AIRACBMSMALLKERNELNET_H_


#include <vector>

#include <torch/torch.h>

struct AirACSKCnnBmNet: torch::nn::Module {
private:
	torch::nn::Conv2d conv0;
//	torch::nn::BatchNorm2d bm0;
	torch::nn::Conv2d conv1;
//	torch::nn::BatchNorm2d bm1;
	torch::nn::Conv2d conv2;
//	torch::nn::BatchNorm2d bm2;
	torch::nn::Conv2d conv3;
//	torch::nn::BatchNorm2d bm3;
	torch::nn::Linear fc1;
	torch::nn::Linear aOut;
	torch::nn::Linear vOut;

	const int actionNum;
public:
	AirACSKCnnBmNet(int aNum);
	~AirACSKCnnBmNet() = default;
	AirACSKCnnBmNet(const AirACSKCnnBmNet&) = delete;

	std::vector<torch::Tensor> forward(torch::Tensor input);
};



#endif /* INC_GYMTEST_AIRNETS_AIRACBMSMALLKERNELNET_H_ */
