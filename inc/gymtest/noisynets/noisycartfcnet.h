/*
 * noisycartfcnet.h
 *
 *  Created on: Aug 31, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_NOISYNETS_NOISYCARTFCNET_H_
#define INC_GYMTEST_NOISYNETS_NOISYCARTFCNET_H_


#include <torch/torch.h>

//#include "netconfig.h"
#include "NoisyLinear.h"

struct NoisyCartFcNet: torch::nn::Module {
private:
	torch::nn::Linear fc0;
//	torch::nn::Linear fcOutput;
//	NoisyLinear fcOutput;
	std::shared_ptr<NoisyLinear> fcOutput;


	const int inputNum;
	const int outputNum;
public:
	NoisyCartFcNet(int intput, int output);
	~NoisyCartFcNet() = default;

	torch::Tensor forward(torch::Tensor input);
	void resetNoise();
};


#endif /* INC_GYMTEST_NOISYNETS_NOISYCARTFCNET_H_ */
