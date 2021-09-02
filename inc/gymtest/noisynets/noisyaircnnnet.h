/*
 * noisyaircnnnet.h
 *
 *  Created on: Aug 31, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_NOISYNETS_NOISYAIRCNNNET_H_
#define INC_GYMTEST_NOISYNETS_NOISYAIRCNNNET_H_


#include <torch/torch.h>

#include "NoisyLinear.h"

struct NoisyAirCnnNet: torch::nn::Module {
private:
	torch::nn::Conv2d conv0;
	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
//	torch::nn::Linear fc2;
//	torch::nn::Linear fc3;
	std::shared_ptr<NoisyLinear> fc2;
	std::shared_ptr<NoisyLinear>  fc3;

public:
	NoisyAirCnnNet(int outputNum);
	~NoisyAirCnnNet() = default;
	NoisyAirCnnNet(const NoisyAirCnnNet&) = delete;

	torch::Tensor forward(torch::Tensor input);
	void resetNoise();
};




#endif /* INC_GYMTEST_NOISYNETS_NOISYAIRCNNNET_H_ */
