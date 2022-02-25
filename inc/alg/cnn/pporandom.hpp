/*
 * pporandom.hpp
 *
 *  Created on: Jul 26, 2021
 *      Author: zf
 */

#ifndef INC_ALG_PPORANDOM_HPP_
#define INC_ALG_PPORANDOM_HPP_


#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/stats.h"
#include "alg/utils/dqnoption.h"

#include "envstep.hpp"

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
class PPORandom: private A2CStoreEnvStep<NetType, EnvType, PolicyType> {
private:
	NetType& bModel;
	EnvType& env;
	EnvType& testEnv;
	PolicyType& policy;
	OptimizerType& optimizer;
	const torch::Device deviceType;
	const at::IntArrayRef inputShape;

	const DqnOption dqnOption;

	std::vector<int64_t> batchInputShape;
	std::vector<int64_t> trajInputShape;

	const int actionNum;

	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("pporamdom");

	AlgTester<NetType, EnvType, PolicyType> tester;

	using A2CStoreEnvStep<NetType, EnvType, PolicyType>::tLogger;

public:
	PPORandom(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, DqnOption option, int actNum);
	~PPORandom() = default;
	PPORandom(const PPORandom& ) = delete;

	void train(const int updateNum);
	void test(const int batchSize, const int epochNum);


	void load();
	void save();
};

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
PPORandom<NetType, EnvType, PolicyType, OptimizerType>::PPORandom(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy,
		OptimizerType& iOptimizer,
		const DqnOption iOption,
		int actNum):
	A2CStoreEnvStep<NetType, EnvType, PolicyType>(iOption),
	bModel(behaviorModel),
	env(iEnv),
	testEnv(tEnv),
	policy(iPolicy),
	optimizer(iOptimizer),
	dqnOption(iOption),
	deviceType(iOption.deviceType),
	inputShape(iOption.inputShape),
	actionNum(actNum),
	tester(behaviorModel, tEnv, iPolicy, iOption, tLogger)
	{
		batchInputShape.push_back(dqnOption.envNum * dqnOption.batchSize);
		trajInputShape.push_back(dqnOption.trajStepNum * dqnOption.envNum);
		for (int i = 1; i < inputShape.size(); i ++) {
			batchInputShape.push_back(inputShape[i]);
			trajInputShape.push_back(inputShape[i]);
		}

//	LOG4CXX_INFO(logger, "indice after initiation \n " << indice);
//	std::srand(unsigned (std::time(0)));
}


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PPORandom<NetType, EnvType, PolicyType, OptimizerType>::train(const int updateNum) {
	LOG4CXX_INFO(logger, "training ");
	load();

	int stepNum = 0;
	int updateIndex = 0;

	this->stateVec = env.reset();
	while (stepNum < updateNum) {
		LOG4CXX_DEBUG(logger, "---------------------------------------> update  " << stepNum);
		this->steps(bModel, env, policy, dqnOption.trajStepNum, stepNum);
		LOG4CXX_DEBUG(logger, "Collect " << dqnOption.trajStepNum << " step samples ");

		//Calculate GAE return
		torch::Tensor lastStateTensor = torch::from_blob(this->stateVec.data(), inputShape).div(dqnOption.inputScale).to(deviceType);
		auto lastRc = bModel.forward(lastStateTensor);
		torch::Tensor lastValueTensor = lastRc[1];
		LOG4CXX_DEBUG(logger, "Get last value " << lastValueTensor.sizes());

		//TODO: Put all tensors in CPU to save storage for inputState tensors?
		at::IntArrayRef batchValueShape{dqnOption.trajStepNum, dqnOption.envNum, 1};

		auto doneData = EnvUtils::FlattenVector(this->donesVec);
		auto rewardData = EnvUtils::FlattenVector(this->rewardsVec);

		//TODO: Check batchValueShape matches original layout
		torch::Tensor valueTensor = torch::stack(this->valuesVec, 0).view(batchValueShape).to(torch::kCPU); //toCPU necessary?
		torch::Tensor rewardTensor = torch::from_blob(rewardData.data(), batchValueShape).div(dqnOption.rewardScale).clamp(dqnOption.rewardMin, dqnOption.rewardMax);
		torch::Tensor doneTensor = torch::from_blob(doneData.data(), batchValueShape);
		LOG4CXX_DEBUG(logger, "dones " << "\n" << doneTensor);
		torch::Tensor gaeReturns = torch::zeros(batchValueShape);
		torch::Tensor returns = torch::zeros(batchValueShape);
		torch::Tensor nextValueTensor = lastValueTensor.to(torch::kCPU).detach();
		torch::Tensor gaeReturn = torch::zeros({dqnOption.envNum, 1});
		torch::Tensor plainReturn = torch::zeros({dqnOption.envNum, 1});
		plainReturn.copy_(nextValueTensor);
		LOG4CXX_DEBUG(logger, "nextValueTensor " << nextValueTensor.sizes());

		LOG4CXX_DEBUG(logger, "--------------------------------------> Calculate GAE");
		for (int i = dqnOption.trajStepNum - 1; i >= 0; i --) {
			torch::Tensor delta = rewardTensor[i] + dqnOption.gamma * nextValueTensor * doneTensor[i] - valueTensor[i];
			gaeReturn = delta + dqnOption.ppoLambda * dqnOption.gamma * gaeReturn * doneTensor[i];

			gaeReturns[i].copy_(gaeReturn);

			if (!dqnOption.tdValue) {
				plainReturn = rewardTensor[i] + dqnOption.gamma * plainReturn * doneTensor[i];
				returns[i].copy_(plainReturn);
			}
			nextValueTensor = valueTensor[i];
		}
		if (dqnOption.tdValue) {
			returns = gaeReturns + valueTensor; //from baseline3.
		}

		//Put all tensors into GPU
		gaeReturns = gaeReturns.to(deviceType).detach();
		returns = returns.to(deviceType).detach();
		LOG4CXX_DEBUG(logger, "Calculated GAE " << gaeReturns.sizes());

		//Calculate old log Pi
		torch::Tensor oldDistTensor = torch::stack(this->pisVec, 0).view({dqnOption.trajStepNum, dqnOption.envNum, actionNum});
		oldDistTensor = torch::softmax(oldDistTensor, -1).to(deviceType);
		LOG4CXX_DEBUG(logger, "oldDistTensor: " << oldDistTensor.sizes());

		auto actionData = EnvUtils::FlattenVector(this->actionsVec);
		torch::Tensor oldActionTensor = torch::from_blob(actionData.data(), {dqnOption.trajStepNum, dqnOption.envNum, 1}, longOpt).to(deviceType);
		LOG4CXX_DEBUG(logger, "oldActionTensor: " << oldActionTensor.sizes());
		torch::Tensor oldPiTensor = oldDistTensor.gather(-1, oldActionTensor).detach();
		LOG4CXX_DEBUG(logger, "oldPiTensor: " << oldPiTensor.sizes());

		//Update
		const int roundNum = dqnOption.trajStepNum / dqnOption.batchSize;

		auto stateData = EnvUtils::FlattenVector(this->statesVec);
		auto stateTensor = torch::from_blob(stateData.data(), trajInputShape).div(dqnOption.inputScale).to(deviceType);
		LOG4CXX_DEBUG(logger, "stateTensor: " << stateTensor.sizes());

		gaeReturns = gaeReturns.view({dqnOption.trajStepNum * dqnOption.envNum, 1});
		oldPiTensor = oldPiTensor.view({dqnOption.trajStepNum * dqnOption.envNum, 1});
		oldActionTensor = oldActionTensor.view({dqnOption.trajStepNum * dqnOption.envNum, 1});
		returns = returns.view({dqnOption.trajStepNum * dqnOption.envNum, 1});


		auto pieceLen = dqnOption.batchSize * dqnOption.envNum;

//		auto indiceTensor = torch::randperm(dqnOption.trajStepNum * dqnOption.envNum, longOpt).view({-1, dqnOption.batchSize * dqnOption.envNum}).to(deviceType);
		for (int epochIndex = 0; epochIndex < dqnOption.epochNum; epochIndex ++) {
			//shuffle index
//			std::random_shuffle(indice.begin(), indice.end());
//			torch::Tensor indiceTensor = torch::from_blob(indice.data(), {indice.size()}, longOpt).to(deviceType);
			auto indiceTensor = torch::randperm(dqnOption.trajStepNum * dqnOption.envNum, longOpt).view({-1, dqnOption.batchSize * dqnOption.envNum}).to(deviceType);

			for (int roundIndex = 0; roundIndex < roundNum; roundIndex ++) {
				updateIndex ++;
				//fetch data
				auto indexPiece = indiceTensor[roundIndex];

				torch::Tensor stateInput = stateTensor.index_select(0, indexPiece);
				torch::Tensor returnPiece = returns.index_select(0, indexPiece);
				torch::Tensor gaePiece = gaeReturns.index_select(0, indexPiece);
				torch::Tensor oldPiPiece = oldPiTensor.index_select(0, indexPiece);
				torch::Tensor oldActionPiece = oldActionTensor.index_select(0, indexPiece);
//				torch::Tensor valuePiece = valueTensor.index_select(0, indexPiece);
				LOG4CXX_DEBUG(logger, "index piece: " << indexPiece);
				LOG4CXX_DEBUG(logger, "state size: " << stateTensor.sizes());
				LOG4CXX_DEBUG(logger, "state input sizes: " << stateInput.sizes());

				auto rc = bModel.forward(stateInput);
				torch::Tensor valueOutput = rc[1];
				torch::Tensor actionOutput = rc[0];

				torch::Tensor valueLossTensor = torch::nn::functional::mse_loss(valueOutput, returnPiece);

				//action loss
				//actionPi and oldPi are logPi
				torch::Tensor advTensor = gaePiece;
				if (dqnOption.normReward) {
					advTensor = (advTensor - advTensor.mean()) / (advTensor.std() + 1e-7);
				}
				torch::Tensor actionPiTensor = torch::softmax(actionOutput, -1);
				torch::Tensor actionPi = actionPiTensor.gather(-1, oldActionPiece);
				torch::Tensor ratio = actionPi / oldPiPiece;
//				LOG4CXX_INFO(logger, "ratio is " << ratio);
				float kl = ratio.mean().to(torch::kCPU).item<float>();
				LOG4CXX_INFO(logger, "actionPiTensor " << actionPiTensor.sizes());
				LOG4CXX_INFO(logger, "actionPi " << actionPi.sizes());
				LOG4CXX_INFO(logger, "oldPiPiece " << oldPiPiece.sizes());
				LOG4CXX_INFO(logger, "ratio " << ratio.sizes());


				auto sur0 = ratio * advTensor.detach();
				auto sur1 = torch::clamp(ratio, 1 - dqnOption.ppoEpsilon, 1 + dqnOption.ppoEpsilon) * advTensor.detach();
				LOG4CXX_DEBUG(logger, "sur1 = " << sur1);
				torch::Tensor actLossTensor = torch::min(sur0, sur1).mean() * (-1);
				LOG4CXX_DEBUG(logger, "actLossTensor = " << actLossTensor);

				//entropy loss
				torch::Tensor actionLogDistTensor = torch::log_softmax(actionOutput, -1);
				LOG4CXX_DEBUG(logger, "actionLogTensor " << actionLogDistTensor);
				torch::Tensor entropyTensor = (-1) * (actionLogDistTensor * actionPiTensor).sum(-1).mean();
				LOG4CXX_DEBUG(logger, "entropy = " << entropyTensor);

				//overall loss
				torch::Tensor lossTensor = actLossTensor + dqnOption.valueCoef * valueLossTensor - dqnOption.entropyCoef * entropyTensor;
//				auto lossTensor = sur0.mean() * (-1);
//				auto lossTensor = actLossTensor - dqnOption.entropyCoef * entropyTensor;

				optimizer.zero_grad();
				lossTensor.backward();
				torch::nn::utils::clip_grad_norm_(bModel.parameters(), dqnOption.maxGradNormClip);
				optimizer.step();

				//print and log
				if ((updateIndex % dqnOption.logInterval) == 0) {
					//print and log
					float lossV = lossTensor.item<float>();
					float vLossV = valueLossTensor.item<float>();
					float aLossV = actLossTensor.item<float>();
					float entropyV = entropyTensor.item<float>();
					float valueV = valueTensor.mean().item<float>();
					LOG4CXX_DEBUG(logger, "loss" << updateIndex << "-" << epochIndex << "-" << roundIndex << ": " << lossV
							<< ", " << vLossV << ", " << aLossV << ", " << entropyV << ", " << kl);

					tLogger.add_scalar("loss/loss", updateIndex, lossV);
					tLogger.add_scalar("loss/vLoss", updateIndex, vLossV);
					tLogger.add_scalar("loss/aLoss", updateIndex, aLossV);
					tLogger.add_scalar("loss/entropy", updateIndex, entropyV);
					tLogger.add_scalar("loss/v", updateIndex, valueV);
					tLogger.add_scalar("loss/kl", updateIndex, kl);
				}
			}
		}

		if ((stepNum % dqnOption.testGapEp) == 0) {
			test(dqnOption.testBatch, dqnOption.testEp);
		}
	}

	save();
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PPORandom<NetType, EnvType, PolicyType, OptimizerType>::test(const int batchSize, const int epochNum) {
	LOG4CXX_INFO(logger, "To test " << epochNum << " episodes");
	if (!dqnOption.toTest) {
		return;
	}

	tester.testAC();
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PPORandom<NetType, EnvType, PolicyType, OptimizerType>::save() {
	if (!dqnOption.saveModel) {
		return;
	}

	AlgUtils::SaveModel(bModel, optimizer, dqnOption.savePathPrefix, logger);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PPORandom<NetType, EnvType, PolicyType, OptimizerType>::load() {
	if (!dqnOption.loadModel) {
		return;
	}

	AlgUtils::LoadModel(bModel, optimizer, dqnOption.loadOptimizer, dqnOption.loadPathPrefix, logger);
}


#endif /* INC_ALG_PPOSHAREDTEST_HPP_ */
