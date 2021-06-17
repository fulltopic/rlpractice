/*
 * a2cnstore.cpp
 *
 *  Created on: May 5, 2021
 *      Author: zf
 */


#include "gymtest/utils/a2cnstore.h"

A2CNStorage::A2CNStorage(torch::Device devType, int size):deviceType (devType), cap(size) {
	//TODO: initial entropy
}


void A2CNStorage::reset() {
	values.clear();
	values.reserve(cap);
	advLogs.clear();
	advLogs.reserve(cap);
	rewards.clear();
	rewards.reserve(cap);
	doneMasks.clear();
	doneMasks.reserve(cap);

	//TODO: clear entropy
//	entropy = torch::zeros({});
	entropy = torch::tensor(0);
}

void A2CNStorage::put(torch::Tensor value, torch::Tensor actLinearOutput,
		torch::Tensor action, torch::Tensor reward, torch::Tensor doneMask) {
	torch::Tensor actionProb = torch::softmax(actLinearOutput, -1);
	torch::Tensor actLogProb = torch::log_softmax(actLinearOutput, -1);
	torch::Tensor advLog = actLogProb.gather(-1, action);
//	LOG4CXX_INFO(logger, "actLogProb: " << actLogProb.sizes());
//	LOG4CXX_INFO(logger, "action: " << action.sizes());

//	if (values.size() == 0) {
//		entropy = -(actionProb * actLogProb).sum(-1).mean();
//	} else {
//		entropy = entropy - (actionProb * actLogProb).sum(-1).mean();
//	}

//	LOG4CXX_INFO(logger, "actLinearOutput: " << actLinearOutput);
//	LOG4CXX_INFO(logger, "actionProb: " << actionProb);
//	LOG4CXX_INFO(logger, "actionLogProb: " << actLogProb);
//	LOG4CXX_INFO(logger, "advLog: " << advLog);
//	auto subEntropy = actionProb * actLogProb;
//	LOG4CXX_INFO(logger, "subEntropy " << subEntropy);
//	auto subSum = subEntropy.sum(-1);
//	LOG4CXX_INFO(logger, "subSum: " << subSum);
//	auto subMean = subSum.mean();
//	LOG4CXX_INFO(logger, "subMean: " << subMean);

	entropy = entropy - (actionProb * actLogProb).sum(-1).mean();
//	LOG4CXX_INFO(logger, "entropy: " << entropy);

	values.push_back(value);
	advLogs.push_back(advLog);
	rewards.push_back(reward);
	doneMasks.push_back(doneMask);
}

//finalValue detached
torch::Tensor A2CNStorage::getLoss(torch::Tensor finalValue, float gamma, float entropyFactor, float valueFactor,
		Stats& stat, LossStats& lossStat) {
//	LOG4CXX_INFO(logger, "getLoss: " << values.size());
	std::vector<torch::Tensor> tmpRs;
	torch::Tensor returnValue = finalValue;
	for (int i = values.size() - 1; i >= 0; i --) {
		tmpRs.push_back(returnValue * gamma * doneMasks[i] + rewards[i]);
		returnValue = tmpRs[tmpRs.size() - 1];
//		LOG4CXX_INFO(logger, "push tmpRs " << tmpRs.size());
	}
//	LOG4CXX_INFO(logger, "To create rs");
	std::vector<torch::Tensor> rs;
	for (int i = tmpRs.size() - 1; i >= 0; i --) {
		rs.push_back(tmpRs[i]);
	}

	torch::Tensor returnTensor = torch::stack(rs, 0).squeeze(-1);
	torch::Tensor valueTensor =  torch::stack(values, 0).squeeze(-1);
//	LOG4CXX_INFO(logger, "return " << returnTensor);
//	LOG4CXX_INFO(logger, "value: " << valueTensor);

	torch::Tensor advTensor = returnTensor - valueTensor;
//	torch::Tensor valueLoss = advTensor.pow(2).mean();
	torch::Tensor valueLoss = torch::nn::functional::mse_loss(valueTensor, returnTensor);
//	LOG4CXX_INFO(logger, "advTensor = " << advTensor);

	torch::Tensor advPi = torch::stack(advLogs, 0).squeeze(-1);
	torch::Tensor actLoss = -(advPi * advTensor.detach()).mean();
//	LOG4CXX_INFO(logger, "advPi = " << advPi);

	entropy = entropy.div(cap);

	torch::Tensor loss = valueFactor * valueLoss + actLoss - entropyFactor * entropy;

	auto lossV = loss.item<float>();
	auto aLossV = actLoss.item<float>();
	auto vLossV = valueLoss.item<float>();
	auto entropyV = entropy.item<float>();

	LOG4CXX_INFO(logger, "loss" << updateSeq << ": " << lossV
		<< ", " << vLossV << ", " << aLossV << ", " << entropyV);
	auto curState = stat.getCurState();
	lossStat.update({lossV, vLossV, aLossV, entropyV,
		vLossV * valueFactor, entropyV * entropyFactor * (-1),
		curState[0], curState[1]});

//	LOG4CXX_INFO(logger, "udpate" << updateSeq << ": " << loss.item<float>() << ", " << valueLoss.item<float>() << ", " << actLoss.item<float>() << ", " << entropy.item<float>());
//	stat.update({loss.item<float>(), valueLoss.item<float>(), actLoss.item<float>(), entropy.item<float>()});

	updateSeq ++;

	return loss;
}
