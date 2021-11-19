/*
 * a3cclienthandle.h
 *
 *  Created on: Nov 7, 2021
 *      Author: zf
 */

#ifndef INC_A3C_A3CCLIENTHANDLE_H_
#define INC_A3C_A3CCLIENTHANDLE_H_

#include <torch/torch.h>

#include <string>
#include <vector>
#include <memory>

#include "a3cconfig.h"

#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

class A3CClient;

class A3CClientHandler {
private:
	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	torch::Tensor cmdTensor = torch::zeros({1}, longOpt);
	torch::Tensor startIndexTensor = torch::zeros({1}, longOpt);
	torch::Tensor endIndexTensor = torch::zeros({1}, longOpt);

	std::vector<torch::Tensor> grads;
	std::vector<torch::Tensor> targetParams;
	std::vector<int> gradSteps;

	std::shared_ptr<A3CClient> client; //TODO: Can shared ptr be released for cycle reference?


	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("a3cchandle");

	void checkGradStep();
	void syncTarget(std::vector<torch::Tensor> params);

public:
	A3CClientHandler(std::vector<at::IntArrayRef> shapes);
	~A3CClientHandler();
	A3CClientHandler(const A3CClientHandler&) = delete;

	void updateGrad(std::vector<torch::Tensor> deltaGrads);
	void sendGrads();

	const std::vector<torch::Tensor>& getTargets();
	void setClient(std::shared_ptr<A3CClient> cp);

	void processRcv(std::vector<torch::Tensor>& ts);
};



#endif /* INC_A3C_A3CCLIENTHANDLE_H_ */
