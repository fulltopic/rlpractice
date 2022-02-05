/*
 * cartacgrunopack.h
 *
 *  Created on: Feb 4, 2022
 *      Author: zf
 */

#ifndef INC_GYMTEST_RNNNETS_LUNARNETS_CARTACGRUNOPACK_H_
#define INC_GYMTEST_RNNNETS_LUNARNETS_CARTACGRUNOPACK_H_


#include <torch/torch.h>
#include <vector>
//#include "netconfig.h"

struct CartACRNNNoPackNet: torch::nn::Module {
private:
	torch::nn::Linear fc0;
	torch::nn::GRU gru1;
	torch::nn::Linear vOutput;
	torch::nn::Linear aOutput;

	const int inputNum;
	const int outputNum;
	const int hiddenNum;

	const torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);

public:
	CartACRNNNoPackNet(int intput, int hiddenSize, int output);
	~CartACRNNNoPackNet() = default;

	//std::vector<torch::Tensor>& states
	std::vector<torch::Tensor> forward(torch::Tensor input, std::vector<long> seqInput);
//	std::vector<torch::Tensor> forward(torch::Tensor input, torch::Tensor shuffleIndex, std::vector<long> seqInput);
	std::vector<torch::Tensor> forward(torch::Tensor input, std::vector<torch::Tensor>& states);
	std::vector<torch::Tensor> forwardTrain(torch::Tensor input);
};


#endif /* INC_GYMTEST_RNNNETS_LUNARNETS_CARTACGRUNOPACK_H_ */
