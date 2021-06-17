/*
 * airsmallkernelnet.h
 *
 *  Created on: May 25, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_AIRNETS_AIRSMALLKERNELNET_H_
#define INC_GYMTEST_AIRNETS_AIRSMALLKERNELNET_H_


#include <vector>

#include <torch/torch.h>

struct AirSKCnnNet: torch::nn::Module {
private:
	torch::nn::Conv2d conv0;
	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Conv2d conv3;
	torch::nn::Linear fc1;
	torch::nn::Linear aOut;

	const int actionNum;
public:
	AirSKCnnNet(int aNum);
	~AirSKCnnNet() = default;
	AirSKCnnNet(const AirSKCnnNet&) = delete;

	torch::Tensor forward(torch::Tensor input);
};




#endif /* INC_GYMTEST_AIRNETS_AIRSMALLKERNELNET_H_ */
