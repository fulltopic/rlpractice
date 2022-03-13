/*
 * airacgrunordnet.h
 *
 *  Created on: Mar 7, 2022
 *      Author: zf
 */

#ifndef INC_GYMTEST_RNNNETS_AIRNETS_AIRACGRUNORDNET_H_
#define INC_GYMTEST_RNNNETS_AIRNETS_AIRACGRUNORDNET_H_


#include <vector>

#include <torch/torch.h>

struct AirACHONRGRUNet: torch::nn::Module {
private:
	torch::nn::Conv2d conv0;
	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::GRU gru0;
//	torch::nn::Linear afc;
//	torch::nn::Linear vfc;
	torch::nn::Linear aOut;
	torch::nn::Linear vOut;

	const int actionNum;
	const int hiddenNum;

	const torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);

public:
	AirACHONRGRUNet(int aNum, int hidden = 512);
	~AirACHONRGRUNet() = default;
	AirACHONRGRUNet(const AirACHONRGRUNet&) = delete;

//	std::vector<torch::Tensor> forward(torch::Tensor input);

//	std::vector<torch::Tensor> forward(torch::Tensor input, std::vector<long> seqInput);
	std::vector<torch::Tensor> forward(torch::Tensor input, std::vector<long> seqInput, std::vector<torch::Tensor> states);
	std::vector<torch::Tensor> forward(torch::Tensor input, std::vector<torch::Tensor>& states);
	std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> forwardNext(torch::Tensor input, std::vector<torch::Tensor> states);
	std::vector<torch::Tensor> forwardNext(torch::Tensor input, int batchSize, std::vector<long> seqInput, std::vector<torch::Tensor> states, torch::Device deviceType);

	std::vector<torch::Tensor> createHStates(const int envNum, torch::Device deviceType);
	void resetHState(const int envIndex, std::vector<torch::Tensor>& states);
	std::vector<torch::Tensor> getHState(const int envIndex, std::vector<torch::Tensor>& states);
};



#endif /* INC_GYMTEST_RNNNETS_AIRNETS_AIRACGRUNORDNET_H_ */
