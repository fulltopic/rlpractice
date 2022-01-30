/*
 * ppo.hpp
 *
 *  Created on: Jun 23, 2021
 *      Author: zf
 */

#ifndef INC_ALG_PPOSHARED_HPP_
#define INC_ALG_PPOSHARED_HPP_


#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <vector>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/stats.h"
#include "alg/utils/dqnoption.h"

#include "envstep.hpp"

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
class PPOShared: private A2CStoreEnvStep<NetType, EnvType, PolicyType> {
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
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("pposhared");

	AlgTester<NetType, EnvType, PolicyType> tester;

//	using A2CStoreEnvStep<NetType, EnvType, PolicyType>::stateVec;
	using A2CStoreEnvStep<NetType, EnvType, PolicyType>::tLogger;

public:
	PPOShared(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, DqnOption option, int actNum);
	~PPOShared() = default;
	PPOShared(const PPOShared& ) = delete;

	void train(const int updateNum);
	void test(const int batchSize, const int epochNum);


	void load();
	void save();
};

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
PPOShared<NetType, EnvType, PolicyType, OptimizerType>::PPOShared(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy,
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
}


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PPOShared<NetType, EnvType, PolicyType, OptimizerType>::train(const int updateNum) {
	LOG4CXX_INFO(logger, "training ");
	load();

	int stepNum = 0;
	int updateIndex = 0;

	this->stateVec = env.reset();
	while (stepNum < updateNum) {
		this->steps(bModel, env, policy, dqnOption.trajStepNum, stepNum);
		LOG4CXX_DEBUG(logger, "Collect " << dqnOption.trajStepNum << " step samples ");

		//Calculate GAE return
		torch::Tensor lastStateTensor = torch::from_blob(this->stateVec.data(), inputShape).div(dqnOption.inputScale).to(deviceType);
		auto lastRc = bModel.forward(lastStateTensor);
		torch::Tensor lastValueTensor = lastRc[1];
		LOG4CXX_DEBUG(logger, "Get last value " << lastValueTensor.sizes());

		//TODO: Put all tensors in CPU to save storage for inputState tensors?
		at::IntArrayRef batchValueShape{dqnOption.trajStepNum, dqnOption.envNum, 1};

//		LOG4CXX_INFO(logger, "dones: " << donesVec.size() << ", " << donesVec[0].size());
//		LOG4CXX_INFO(logger, "dones numel: " << EnvUtils::FlattenVector(donesVec).size());
		auto doneData = EnvUtils::FlattenVector(this->donesVec);
		auto rewardData = EnvUtils::FlattenVector(this->rewardsVec);
//		LOG4CXX_INFO(logger, "done raw data ");

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
		//tmp solution
		torch::Tensor returnDoneTensor = torch::zeros(batchValueShape);
		returnDoneTensor.copy_(doneTensor);
		for (int i = dqnOption.trajStepNum - 1; i >= 0; i --) {
//			LOG4CXX_INFO(logger, "gae before: " << gaeReturn);
			torch::Tensor delta = rewardTensor[i] + dqnOption.gamma * nextValueTensor * doneTensor[i] - valueTensor[i];
			gaeReturn = delta + dqnOption.ppoLambda * dqnOption.gamma * gaeReturn * doneTensor[i];
//			LOG4CXX_INFO(logger, "reward: " << rewardTensor[i]);
//			LOG4CXX_INFO(logger, "done: " << doneTensor[i]);
//			LOG4CXX_INFO(logger, "nextValueTensor: "<< nextValueTensor);
//			LOG4CXX_INFO(logger, "valueTensor: " << valueTensor[i]);
//			LOG4CXX_INFO(logger, "delta: " << delta);
//			LOG4CXX_INFO(logger, "gaeReturn after: " << gaeReturn);
			plainReturn = rewardTensor[i] + dqnOption.gamma * plainReturn * doneTensor[i];

			gaeReturns[i].copy_(gaeReturn);
			returns[i].copy_(plainReturn);
			nextValueTensor = valueTensor[i];
		}

		//Put all tensors into GPU
		//normalize both gae and plain return
		gaeReturns = ((gaeReturns - gaeReturns.mean()) / (gaeReturns.std() + 1e-7)).to(deviceType).detach();
		LOG4CXX_DEBUG(logger, "Calculated GAE " << gaeReturns.sizes());
		//No normalization on return
		returns = returns.to(deviceType).detach();
//		returns = ((returns - returns.mean()) / (returns.std() + 1e-7)).to(deviceType).detach();
//		LOG4CXX_DEBUG(logger, "Calculated plain return " << returns.sizes());

		//Calculate old log Pi
		torch::Tensor oldDistTensor = torch::stack(this->pisVec, 0).view({dqnOption.trajStepNum, dqnOption.envNum, actionNum});
		oldDistTensor = torch::log_softmax(oldDistTensor, -1).to(deviceType);
		LOG4CXX_DEBUG(logger, "oldDistTensor: " << oldDistTensor.sizes());

		auto actionData = EnvUtils::FlattenVector(this->actionsVec);
		torch::Tensor oldActionTensor = torch::from_blob(actionData.data(), {dqnOption.trajStepNum, dqnOption.envNum, 1}, longOpt).to(deviceType);
//		torch::Tensor testActionTensor = torch::zeros({dqnOption.trajStepNum, dqnOption.envNum, 1}, longOpt).to(deviceType);
		LOG4CXX_DEBUG(logger, "oldActionTensor: " << oldActionTensor.sizes());
		torch::Tensor oldPiTensor = oldDistTensor.gather(-1, oldActionTensor).detach();
		LOG4CXX_DEBUG(logger, "oldPiTensor: " << oldPiTensor.sizes());


		const int roundNum = dqnOption.trajStepNum / dqnOption.batchSize;

		auto stateData = EnvUtils::FlattenVector(this->statesVec);
		auto stateTensor = torch::from_blob(stateData.data(), trajInputShape).div(dqnOption.inputScale).to(deviceType);
		LOG4CXX_DEBUG(logger, "stateTensor: " << stateTensor.sizes());
		auto stateTensors = torch::chunk(stateTensor, roundNum, 0);
		//TODO: Try view gaeReturns before chunk to avoid view for each slice
		auto gaeTensors = torch::chunk(gaeReturns, roundNum, 0);
		for (int i = 0; i < roundNum; i ++) {
			gaeTensors[i] = gaeTensors[i].view({dqnOption.batchSize * dqnOption.envNum, 1});
		}
		auto oldPiTensors = torch::chunk(oldPiTensor, roundNum, 0); //oldPiTensor = {trajStep, envNum, 1}
		for (int i = 0; i < roundNum; i ++) {
			oldPiTensors[i] = oldPiTensors[i].view({dqnOption.batchSize * dqnOption.envNum, 1});
		}
		auto actionTensors = torch::chunk(oldActionTensor, roundNum, 0);
		for (int i = 0; i < roundNum; i ++) {
			actionTensors[i] = actionTensors[i].view({dqnOption.batchSize * dqnOption.envNum, 1});
		}
		auto returnTensors = torch::chunk(returns, roundNum, 0);
		for (int i = 0; i < roundNum; i ++) {
			returnTensors[i] = returnTensors[i].view({dqnOption.batchSize * dqnOption.envNum, 1});
		}
		valueTensor = valueTensor.to(deviceType).detach();
		auto valueTensors = torch::chunk(valueTensor, roundNum, 0);
		for (int i = 0; i < roundNum; i ++) {
			valueTensors[i] = valueTensors[i].view({dqnOption.batchSize * dqnOption.envNum, 1});
		}

		std::vector<bool> toUpdate(roundNum, true);
		for (int epochIndex = 0; epochIndex < dqnOption.epochNum; epochIndex ++) {
			for (int roundIndex = 0; roundIndex < roundNum; roundIndex ++) {
				updateIndex ++;

				if (!toUpdate[roundIndex]) {
					LOG4CXX_DEBUG(logger, "-----------------------------------> early stop " << roundIndex);
					continue;
				} else {
					LOG4CXX_DEBUG(logger, "-------------------------------------> update one batch " << roundIndex);
				}

				LOG4CXX_DEBUG(logger, "train input " << stateTensors[roundIndex].sizes());
				auto rc = bModel.forward(stateTensors[roundIndex]);
				torch::Tensor valueOutput = rc[1];
				torch::Tensor actionOutput = rc[0];
				LOG4CXX_DEBUG(logger, "value output " << valueOutput);
				LOG4CXX_DEBUG(logger, " actionOutput " << actionOutput);

				torch::Tensor valueLossTensor = torch::nn::functional::mse_loss(valueOutput, returnTensors[roundIndex]);

				//action loss
				//actionPi and oldPi are logPi
				torch::Tensor advTensor = gaeTensors[roundIndex];
				LOG4CXX_DEBUG(logger, "advTensor: " << advTensor);
				torch::Tensor actionLogDistTensor = torch::log_softmax(actionOutput, -1);
				LOG4CXX_DEBUG(logger, "actionDistTensor: " << actionLogDistTensor);
				LOG4CXX_DEBUG(logger, "actionTensors[round] = " << actionTensors[roundIndex].sizes());
				torch::Tensor actionPi = actionLogDistTensor.gather(-1, actionTensors[roundIndex]);
				LOG4CXX_DEBUG(logger, "actionPi: " << actionPi);
				auto ratio = torch::exp(actionPi - oldPiTensors[roundIndex]);
				LOG4CXX_DEBUG(logger, "ratio = " << ratio);


				auto piDelta = actionPi - oldPiTensors[roundIndex];
				float kl = ((torch::exp(piDelta) - 1) - piDelta).mean().abs().to(torch::kCPU).item<float>();
				if (dqnOption.klEarlyStop) {
					if (kl > dqnOption.maxKl) {
						LOG4CXX_INFO(logger, "Early stop: " << kl);
						toUpdate[roundIndex] = false;
						continue;
					}
				}

				auto sur0 = ratio * advTensor;
				LOG4CXX_DEBUG(logger, "sur0 = " << sur0);
				auto sur1 = torch::clamp(ratio, 1 - dqnOption.ppoEpsilon, 1 + dqnOption.ppoEpsilon) * advTensor;
				LOG4CXX_DEBUG(logger, "sur1 = " << sur1);
				torch::Tensor actLossTensor = torch::min(sur0, sur1).mean() * (-1);
				LOG4CXX_DEBUG(logger, "actLossTensor = " << actLossTensor);

				//entropy loss
				torch::Tensor actionPiTensor = torch::softmax(actionOutput, -1);
				LOG4CXX_DEBUG(logger, "actionLogTensor " << actionPiTensor);
				torch::Tensor entropyTensor = (-1) * (actionLogDistTensor * actionPiTensor).sum(-1).mean();
				LOG4CXX_DEBUG(logger, "entropy = " << entropyTensor);

				torch::Tensor lossTensor = actLossTensor + dqnOption.valueCoef * valueLossTensor - dqnOption.entropyCoef * entropyTensor;

//				torch::Tensor lossTensor = actLossTensor;

				optimizer.zero_grad();
				lossTensor.backward();
				torch::nn::utils::clip_grad_norm_(bModel.parameters(), dqnOption.maxGradNormClip);
				optimizer.step();

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
void PPOShared<NetType, EnvType, PolicyType, OptimizerType>::test(const int batchSize, const int epochNum) {
	tester.testAC();
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PPOShared<NetType, EnvType, PolicyType, OptimizerType>::save() {
	if (!dqnOption.saveModel) {
		return;
	}

	AlgUtils::SaveModel(bModel, optimizer, dqnOption.savePathPrefix, logger);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PPOShared<NetType, EnvType, PolicyType, OptimizerType>::load() {
	if (!dqnOption.loadModel) {
		return;
	}

	AlgUtils::LoadModel(bModel, optimizer, dqnOption.loadOptimizer, dqnOption.loadPathPrefix, logger);
}

#endif /* INC_ALG_PPOSHARED_HPP_ */
