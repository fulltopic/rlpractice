/*
 * cartacrnnnet.h
 *
 *  Created on: Jan 27, 2022
 *      Author: zf
 */

#ifndef INC_GYMTEST_RNNNETS_LUNARNETS_CARTACRNNNET_H_
#define INC_GYMTEST_RNNNETS_LUNARNETS_CARTACRNNNET_H_


#include <torch/torch.h>
#include <vector>
//#include "netconfig.h"

struct CartACRNNFcNet: torch::nn::Module {
private:
	torch::nn::GRU gru0;
	torch::nn::Linear vOutput;
	torch::nn::Linear aOutput;

	const int inputNum;
	const int outputNum;
	const int hiddenNum;

	const torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);

public:
	CartACRNNFcNet(int intput, int hiddenSize, int output);
	~CartACRNNFcNet() = default;

	//std::vector<torch::Tensor>& states
	std::vector<torch::Tensor> forward(torch::Tensor input, std::vector<long> seqInput);
	std::vector<torch::Tensor> forward(torch::Tensor input, torch::Tensor shuffleIndex, std::vector<long> seqInput);
	std::vector<torch::Tensor> forward(torch::Tensor input, std::vector<torch::Tensor>& states);

	std::vector<torch::Tensor> createHStates(const int envNum, torch::Device deviceType);
};



#endif /* INC_GYMTEST_RNNNETS_LUNARNETS_CARTACRNNNET_H_ */
