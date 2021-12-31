/*
 * priodqn.hpp
 *
 *  Created on: Aug 26, 2021
 *      Author: zf
 */

#ifndef INC_ALG_PRIODQN_HPP_
#define INC_ALG_PRIODQN_HPP_

#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/stats.h"
//#include "gymtest/utils/lossstats.h"

//#include "gymtest/utils/mheap.hpp"
//#include "gymtest/utils/stree.h"
#include "utils/priorb.h"
#include "utils/algtester.hpp"
#include "utils/utils.hpp"
#include "dqnoption.h"


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
class PrioDqn {
private:
	NetType& bModel;
	NetType& tModel;
	EnvType& env;
	EnvType& testEnv;
	PolicyType& policy;
	OptimizerType& optimizer;
	const torch::Device deviceType;
	const at::IntArrayRef inputShape;

	const DqnOption dqnOption;

	uint32_t updateNum = 0;

	float beta;
	float maxAveReward;

	const torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("priodqn");
	TensorBoardLogger tLogger;

	PrioReplayBuffer buffer; //buffer has to be defined after dqnOption so ReplayBuffer can get all parameters of dqnOption.

	//Declare at last to be initialized after tLogger
	AlgTester<NetType, EnvType, PolicyType> tester;


	void updateModel(bool force = false);
	void updateStep(const float epochNum);

	void load();
	void save();

	void saveByReward(float reward);


public:
	PrioDqn(NetType& iModel, NetType& iTModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, DqnOption iOption);
	~PrioDqn() = default;
	PrioDqn(const PrioDqn&) = delete;

	void train(const int epochNum);
	void test(const int epochNum, bool render = false);
};


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
PrioDqn<NetType, EnvType, PolicyType, OptimizerType>::PrioDqn(NetType& iModel, NetType& iTModel,
		EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer,
		DqnOption iOption):
	bModel(iModel),
	tModel(iTModel),
	env(iEnv),
	testEnv(tEnv),
	policy(iPolicy),
	optimizer(iOptimizer),
	dqnOption(iOption),
	deviceType(iOption.deviceType),
	inputShape(iOption.inputShape),
	buffer(iOption.rbCap, iOption.inputShape, iOption.pbEpsilon),
	beta(iOption.pbBetaBegin),
	maxAveReward(iOption.saveThreshold),
	tLogger(iOption.tensorboardLogPath.c_str()),
	tester(iTModel, tEnv, iPolicy, iOption, tLogger)
{

}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PrioDqn<NetType, EnvType, PolicyType, OptimizerType>::train(const int epochNum) {
	load();
	updateModel(true); //model assignment
//	tModel.eval();

	//only one env
	std::vector<float> statRewards(dqnOption.envNum, 0);
	std::vector<float> statLens(dqnOption.envNum, 0);
	std::vector<float> statSumRewards(dqnOption.envNum, 0);
	std::vector<float> statSumLens(dqnOption.envNum, 0);
	std::vector<int> livePerEp(dqnOption.envNum, 0);
	int epNum = 0;

	std::vector<float> stateVec = env.reset();
	while (updateNum < epochNum) {
		for (int k = 0; k < dqnOption.envStep; k ++) {
			updateNum ++;
			LOG4CXX_DEBUG(logger, "---------------------------------------> update " << updateNum);

			//Run step
			torch::Tensor cpuInputTensor = torch::from_blob(stateVec.data(), inputShape);
			torch::Tensor inputTensor = cpuInputTensor.to(deviceType).div(dqnOption.inputScale);

			torch::Tensor outputTensor = bModel.forward(inputTensor); //TODO: bModel or tModel?
			LOG4CXX_DEBUG(logger, "outputTensor: " << outputTensor);
			std::vector<int64_t> actions = policy.getActions(outputTensor);

			auto stepResult = env.step(actions);
			auto nextInputVec = std::get<0>(stepResult);
			auto rewardVec = std::get<1>(stepResult);
			auto doneVec = std::get<2>(stepResult);

			Stats::UpdateReward(statRewards, rewardVec);
			Stats::UpdateLen(statLens);

			float doneMask = 1;
			if (doneVec[0]) {
				doneMask = 0;

				tLogger.add_scalar("train/len", updateNum, statLens[0]);
				tLogger.add_scalar("train/reward", updateNum, statRewards[0]);
				LOG4CXX_INFO(logger, "" << policy.getEpsilon() << "--" << updateNum << ", " << statLens[0] << ", " << statRewards[0]);

				if (dqnOption.multiLifes) {
					livePerEp[0] ++;
					statSumRewards[0] += statRewards[0];
					statSumLens[0] += statLens[0];

					if (livePerEp[0] >= dqnOption.donePerEp) {
						epNum ++;

						tLogger.add_scalar("train/sumLen", epNum, statSumLens[0]);
						tLogger.add_scalar("train/sumReward", epNum, statSumRewards[0]);

						statSumLens[0] = 0;
						statSumRewards[0] = 0;
						livePerEp[0] = 0;
					}
				}

				statRewards[0] = 0;
				statLens[0] = 0;
			}

			torch::Tensor nextInputTensor = torch::from_blob(nextInputVec.data(), inputShape);
			float reward = std::max(std::min((rewardVec[0] / dqnOption.rewardScale), dqnOption.rewardMax), dqnOption.rewardMin);
			buffer.add(cpuInputTensor, nextInputTensor, actions[0], reward, doneMask);

			//Update
			stateVec = nextInputVec;
			updateStep(epochNum);
		}

		//Learning
		if (updateNum < dqnOption.startStep) {
			continue;
		}

//		int sampleNum = buffer.size();
		auto rc = buffer.getSampleIndex(dqnOption.batchSize);
		torch::Tensor sampleIndice = std::get<0>(rc);
		torch::Tensor samplePrios = std::get<1>(rc);
//		LOG4CXX_INFO(logger, "sample index: \n" << sampleIndice);
//		LOG4CXX_INFO(logger, "sample prios: \n" << samplePrios);

		torch::Tensor curStateTensor = buffer.states.index_select(0, sampleIndice).to(deviceType).to(torch::kFloat).div(dqnOption.inputScale);
		torch::Tensor actionTensor = buffer.actions.index_select(0, sampleIndice).to(deviceType).to(torch::kLong);
		torch::Tensor rewardTensor = buffer.rewards.index_select(0, sampleIndice).to(deviceType).to(torch::kFloat);
		torch::Tensor doneMaskTensor = buffer.donesMask.index_select(0, sampleIndice).to(deviceType).to(torch::kFloat);
		LOG4CXX_DEBUG(logger, "sampleIndex before: " << sampleIndice);
		torch::Tensor nextSampleIndice = (sampleIndice + 1) % dqnOption.rbCap;
		torch::Tensor nextStateTensor = buffer.states.index_select(0, nextSampleIndice).to(deviceType).to(torch::kFloat).div(dqnOption.inputScale);
		LOG4CXX_DEBUG(logger, "sampleIndice after: " << nextSampleIndice);
		LOG4CXX_DEBUG(logger, "nextStateTensor: " << nextStateTensor.sizes());

		torch::Tensor targetQ;
		LOG4CXX_DEBUG(logger, "targetQ before " << targetQ);
		{
			torch::NoGradGuard guard;

			torch::Tensor nextBOutput = bModel.forward(nextStateTensor).detach();
			torch::Tensor nextTOutput = tModel.forward(nextStateTensor).detach();
			torch::Tensor maxActions = nextBOutput.argmax(-1).unsqueeze(-1);
			torch::Tensor nextQ = nextTOutput.gather(-1, maxActions);
			targetQ = rewardTensor + dqnOption.gamma * nextQ * doneMaskTensor;
			targetQ = targetQ.detach();
			LOG4CXX_DEBUG(logger, "nextBOutput: " << nextBOutput);
			LOG4CXX_DEBUG(logger, "maxActions: " << maxActions);
			LOG4CXX_DEBUG(logger, "nextTOutput: " << nextTOutput);
			LOG4CXX_DEBUG(logger, "nextQ: " << nextQ);
			LOG4CXX_DEBUG(logger, "rewardTensor: " << rewardTensor);
			LOG4CXX_DEBUG(logger, "doneMaskTensor: " << doneMaskTensor);
			LOG4CXX_DEBUG(logger, "targetQ: " << targetQ);
		}

		torch::Tensor curOutput = bModel.forward(curStateTensor);
		LOG4CXX_DEBUG(logger, "curOutput: " << curOutput);
		torch::Tensor curQ = curOutput.gather(-1, actionTensor); //TODO: shape of actionTensor and curQ
		LOG4CXX_DEBUG(logger, "curQ: " << curQ);

		//PRIO
		torch::Tensor probs = samplePrios.view({dqnOption.batchSize, 1}).to(deviceType) / buffer.sum();
		LOG4CXX_DEBUG(logger, "Sum: " << buffer.sum());
		LOG4CXX_DEBUG(logger, "probs: " << probs);
		torch::Tensor weights = (buffer.size() * probs).pow(- beta);
		weights = (weights / weights.max()).sqrt();
//		torch::Tensor lossTensor = weights * ((targetQ - curQ).pow(2) / 2.0f);
		torch::Tensor lossTensor = torch::nn::functional::mse_loss(weights * curQ, weights * targetQ);
//		torch::Tensor loss = (weights * ((targetQ - curQ).pow(2) / 2.0f)).mean();
		torch::Tensor loss = lossTensor.mean();
		LOG4CXX_DEBUG(logger, "weights: " << weights);
		LOG4CXX_DEBUG(logger, "loss: " << loss);

		torch::Tensor delta = (targetQ - curQ).abs().detach();

		optimizer.zero_grad();
		loss.backward();
		torch::nn::utils::clip_grad_norm_(bModel.parameters(), dqnOption.maxGradNormClip);
		optimizer.step();

		//TODO: prios calculate and update
		torch::Tensor newPrios = delta.pow(dqnOption.pbAlpha) + dqnOption.pbEpsilon;
		LOG4CXX_DEBUG(logger, "delta: " << delta);
//		LOG4CXX_INFO(logger, "newPrios: \n" << newPrios);
		buffer.update(sampleIndice, newPrios);

		if ((updateNum % dqnOption.logInterval) == 0) {
			float deltaValue = delta.mean().item<float>();
			float weightValue = weights.mean().item<float>();
			float lossValue = loss.item<float>();
			float curQValue = curQ.mean().item<float>();

			tLogger.add_scalar("loss/loss", updateNum, lossValue);
			tLogger.add_scalar("loss/q", updateNum, curQValue);
			tLogger.add_scalar("loss/weight", updateNum, weightValue);
			tLogger.add_scalar("loss/delta", updateNum, deltaValue);
			tLogger.add_scalar("loss/beta", updateNum, beta);
			tLogger.add_scalar("loss/epsilon", updateNum, policy.getEpsilon());
		}

		if (dqnOption.toTest) {
			if (updateNum % dqnOption.testGapEp == 0) {
				test(false, false);
			}
		}
	}

	save();
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PrioDqn<NetType, EnvType, PolicyType, OptimizerType>::updateModel(bool force) {
	if (!force) {
		if ((updateNum % dqnOption.targetUpdateStep) != 0) {
			return;
		}
	}

	AlgUtils::SyncNet(bModel, tModel, dqnOption.tau);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PrioDqn<NetType, EnvType, PolicyType, OptimizerType>::updateStep(const float epochNum) {
	updateModel(false);

	if (updateNum <= dqnOption.pbBetaPart * epochNum) {
		beta = dqnOption.pbBetaBegin + (updateNum / (dqnOption.pbBetaPart * epochNum)) * (dqnOption.pbBetaEnd - dqnOption.pbBetaBegin);
	}

	if (updateNum <= (dqnOption.explorePart * epochNum)) {
		float newEpsilon = (dqnOption.exploreBegin - dqnOption.exploreEnd) * (epochNum * dqnOption.explorePart - updateNum) / (epochNum * dqnOption.explorePart) + dqnOption.exploreEnd;
		policy.updateEpsilon(newEpsilon);
	}

}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PrioDqn<NetType, EnvType, PolicyType, OptimizerType>::test(const int epochNum, bool render) {
	tester.testPlain();
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PrioDqn<NetType, EnvType, PolicyType, OptimizerType>::save() {
	if (!dqnOption.saveModel) {
		return;
	}

	AlgUtils::SaveModel(bModel, optimizer, dqnOption.savePathPrefix, logger);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PrioDqn<NetType, EnvType, PolicyType, OptimizerType>::load() {
	if (!dqnOption.loadModel) {
		return;
	}

	AlgUtils::LoadModel(bModel, optimizer, dqnOption.loadOptimizer, dqnOption.loadPathPrefix, logger);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PrioDqn<NetType, EnvType, PolicyType, OptimizerType>::saveByReward(float reward) {
	if (!dqnOption.saveModel) {
		return;
	}

	std::string path = dqnOption.savePathPrefix + "_" + std::to_string(reward);
	AlgUtils::SaveModel(tModel, optimizer, path, logger);
}


#endif /* INC_ALG_PRIODQN_HPP_ */
