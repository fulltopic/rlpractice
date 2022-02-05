/*
 * cartaclstmnet.h
 *
 *  Created on: Feb 3, 2022
 *      Author: zf
 */

#ifndef INC_GYMTEST_RNNNETS_LUNARNETS_CARTACLSTMNET_H_
#define INC_GYMTEST_RNNNETS_LUNARNETS_CARTACLSTMNET_H_

#include <torch/torch.h>
#include <vector>

struct CartACLSTMFcNet: torch::nn::Module {
private:
	torch::nn::LSTM lstm0;
	torch::nn::Linear vOutput;
	torch::nn::Linear aOutput;

	const int inputNum;
	const int outputNum;
	const int hiddenNum;

	const torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);

public:
	CartACLSTMFcNet(int intput, int hiddenSize, int output);
	~CartACLSTMFcNet() = default;

	//std::vector<torch::Tensor>& states
	std::vector<torch::Tensor> forward(torch::Tensor input, std::vector<long> seqInput);
	std::vector<torch::Tensor> forward(torch::Tensor input, std::vector<torch::Tensor>& cx, std::vector<torch::Tensor>& hx);
};



#endif /* INC_GYMTEST_RNNNETS_LUNARNETS_CARTACLSTMNET_H_ */
