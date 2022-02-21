/*
 * cartacgruslim.h
 *
 *  Created on: Feb 14, 2022
 *      Author: zf
 */

#ifndef INC_GYMTEST_RNNNETS_LUNARNETS_CARTACGRUSLIM_H_
#define INC_GYMTEST_RNNNETS_LUNARNETS_CARTACGRUSLIM_H_


#include <torch/torch.h>
#include <vector>
//#include "netconfig.h"

struct CartACGRUTruncFcSlimNet: torch::nn::Module {
private:
	torch::nn::GRU gru0;
	torch::nn::Linear vOutput;
	torch::nn::Linear aOutput;

	const int inputNum;
	const int outputNum;
	const int hiddenNum;

	const torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);

public:
	CartACGRUTruncFcSlimNet(int intput, int hiddenSize, int output);
	~CartACGRUTruncFcSlimNet() = default;

	//std::vector<torch::Tensor>& states
	std::vector<torch::Tensor> forward(torch::Tensor input, std::vector<long> seqInput);
	std::vector<torch::Tensor> forward(torch::Tensor input, int batchSize, int seqInput, std::vector<torch::Tensor>& states, torch::Device deviceType);
	std::vector<torch::Tensor> forward(torch::Tensor input, std::vector<torch::Tensor>& states);

	std::vector<torch::Tensor> createHStates(const int envNum, torch::Device deviceType);
	void resetHState(const int envIndex, std::vector<torch::Tensor>& states);
	std::vector<torch::Tensor> getHState(const int envIndex, std::vector<torch::Tensor>& states);
};



#endif /* INC_GYMTEST_RNNNETS_LUNARNETS_CARTACGRUSLIM_H_ */
