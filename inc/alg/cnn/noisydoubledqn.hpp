/*
 * noisydoubledqn.hpp
 *
 *  Created on: Sep 1, 2021
 *      Author: zf
 */

#ifndef INC_ALG_NOISYDOUBLEDQN_HPP_
#define INC_ALG_NOISYDOUBLEDQN_HPP_


#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <tensorboard_logger.h>


#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/stats.h"
//#include "gymtest/utils/lossstats.h"
#include "alg/utils/utils.hpp"
#include "alg/utils/algtester.hpp"
#include "alg/utils/dqnoption.h"
#include "alg/utils/replaybuffer.h"

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
class NoisyDoubleDqn {
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
	bool startTraining = false;
	float maxAveReward;


	const torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("dqn");
	TensorBoardLogger tLogger;


	ReplayBuffer buffer; //buffer has to be defined after dqnOption so ReplayBuffer can get all parameters of dqnOption.
	AlgTester<NetType, EnvType, PolicyType> tester;

	void updateModel(bool force = false);
	void updateStep(const float epochNum);

	void load();
	void save();
	void saveByReward(float reward);

public:
	NoisyDoubleDqn(NetType& iModel, NetType& iTModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, DqnOption iOption);
	~NoisyDoubleDqn() = default;
	NoisyDoubleDqn(const NoisyDoubleDqn&) = delete;

	void train(const int epochNum);
	void test(const int epochNum, bool render = false);
};


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
NoisyDoubleDqn<NetType, EnvType, PolicyType, OptimizerType>::NoisyDoubleDqn(NetType& iModel, NetType& iTModel,
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
	maxAveReward(iOption.saveThreshold),
	buffer(iOption.rbCap, iOption.inputShape),
	tLogger(iOption.tensorboardLogPath.c_str()),
	tester(iTModel, tEnv, iPolicy, iOption, tLogger)
{
	maxAveReward = iOption.saveThreshold;
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void NoisyDoubleDqn<NetType, EnvType, PolicyType, OptimizerType>::train(const int epochNum) {
	load();
	updateModel(true); //model assignment

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
			torch::NoGradGuard guard;

//			torch::autograd::AnomalyMode::set_enabled(true);
			bModel.resetNoise(); //v1, v2
//			tModel.resetNoise(); //v1

			//Run step
			torch::Tensor cpuinputTensor = torch::from_blob(stateVec.data(), inputShape);
			torch::Tensor inputTensor = cpuinputTensor.to(deviceType).div(dqnOption.inputScale);

//			bModel.resetNoise();
			torch::Tensor outputTensor = bModel.forward(inputTensor); //TODO: bModel or tModel?
			std::vector<int64_t> actions = policy.getTestActions(outputTensor);
			LOG4CXX_DEBUG(logger, "inputTensor: " << inputTensor);
			LOG4CXX_DEBUG(logger, "outputTensor: " << outputTensor);
			LOG4CXX_DEBUG(logger, "actions: " << actions);

			auto stepResult = env.step(actions);
			auto nextInputVec = std::get<0>(stepResult);
			auto rewardVec = std::get<1>(stepResult);
			auto doneVec = std::get<2>(stepResult);
			LOG4CXX_DEBUG(logger, "reward: " << rewardVec);

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
			buffer.add(cpuinputTensor, nextInputTensor, actions[0], reward, doneMask);

			//Update
			stateVec = nextInputVec;
			updateStep(epochNum);
		}


		//Learning
		if (updateNum < dqnOption.startStep) {
			continue;
		}

		torch::Tensor sampleIndice = buffer.getSampleIndex(dqnOption.batchSize);
		torch::Tensor curStateTensor = buffer.states.index_select(0, sampleIndice).to(deviceType).to(torch::kFloat).div(dqnOption.inputScale);
		torch::Tensor actionTensor = buffer.actions.index_select(0, sampleIndice).to(deviceType).to(torch::kLong);
		torch::Tensor rewardTensor = buffer.rewards.index_select(0, sampleIndice).to(deviceType).to(torch::kFloat);
		torch::Tensor doneMaskTensor = buffer.donesMask.index_select(0, sampleIndice).to(deviceType).to(torch::kFloat);
		LOG4CXX_DEBUG(logger, "rewardTensor: " << rewardTensor);
		LOG4CXX_DEBUG(logger, "actionTensor: " << actionTensor);

		LOG4CXX_DEBUG(logger, "sampleIndex before: " << sampleIndice);
		sampleIndice = (sampleIndice + 1) % dqnOption.rbCap;
		torch::Tensor nextStateTensor = buffer.states.index_select(0, sampleIndice).to(deviceType).to(torch::kFloat).div(dqnOption.inputScale);
		LOG4CXX_DEBUG(logger, "sampleIndice after: " << sampleIndice);
		LOG4CXX_DEBUG(logger, "nextStateTensor: " << nextStateTensor);

		bModel.resetNoise();
		tModel.resetNoise();

		torch::Tensor targetQ;
		LOG4CXX_DEBUG(logger, "targetQ before " << targetQ);
		{
			torch::NoGradGuard guard;

//			tModel.resetNoise(); //v2
			torch::Tensor nextBOutput = bModel.forward(nextStateTensor).detach();
			torch::Tensor nextTOutput = tModel.forward(nextStateTensor).detach();
			torch::Tensor maxActions = nextBOutput.argmax(-1).unsqueeze(-1);
//			torch::Tensor maxActions = std::get<1>(nextBOutput.max(-1));
//			maxActions = maxActions.unsqueeze(-1);
			LOG4CXX_DEBUG(logger, "maxActions: " << maxActions);
			torch::Tensor nextQ = nextTOutput.gather(-1, maxActions);
//			torch::Tensor
			targetQ = rewardTensor + dqnOption.gamma * nextQ * doneMaskTensor;
			targetQ = targetQ.detach();
			LOG4CXX_DEBUG(logger, "nextBOutput: " << nextBOutput);
			LOG4CXX_DEBUG(logger, "nextTOutput: " << nextTOutput);
			LOG4CXX_DEBUG(logger, "nextQ: " << nextQ);
			LOG4CXX_DEBUG(logger, "rewardTensor: " << rewardTensor);
			LOG4CXX_DEBUG(logger, "doneMaskTensor: " << doneMaskTensor);
			LOG4CXX_DEBUG(logger, "targetQ: " << targetQ);
		}

		//		bModel.resetNoise(); //v2
		torch::Tensor curOutput = bModel.forward(curStateTensor);
		LOG4CXX_DEBUG(logger, "curOutput: " << curOutput);
		torch::Tensor curQ = curOutput.gather(-1, actionTensor); //TODO: shape of actionTensor and curQ
		LOG4CXX_DEBUG(logger, "curQ: " << curQ);


		optimizer.zero_grad();

//		auto loss = torch::nn::functional::smooth_l1_loss(curQ, targetQ);
		auto loss = torch::nn::functional::mse_loss(curQ, targetQ);
		LOG4CXX_DEBUG(logger, "loss " << loss);


		loss.backward();
		torch::nn::utils::clip_grad_norm_(bModel.parameters(), dqnOption.maxGradNormClip);
		optimizer.step();

//		torch::autograd::AnomalyMode::set_enabled(false);
		if ((updateNum % dqnOption.logInterval) == 0) {
			torch::NoGradGuard guard;

			float lossValue = loss.item<float>();
			float qValue = curQ.mean().item<float>();
			tLogger.add_scalar("loss/loss", updateNum, lossValue);
			tLogger.add_scalar("loss/q", updateNum, qValue);
//			tLogger.add_scalar("loss/epsilon", updateNum, policy.getEpsilon());
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
void NoisyDoubleDqn<NetType, EnvType, PolicyType, OptimizerType>::updateModel(bool force) {
	if (!force) {
		if ((updateNum % dqnOption.targetUpdateStep) != 0) {
			return;
		}
	}

	AlgUtils::SyncNet(bModel, tModel, dqnOption.tau);
//	LOG4CXX_INFO(logger, "----------------------------------------> target network synched");
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void NoisyDoubleDqn<NetType, EnvType, PolicyType, OptimizerType>::updateStep(const float epochNum) {
//	if (!startTraining) {
//		if (updateNum >= dqnOption.startStep) {
//			updateNum = 0;
//			startTraining = true;
//		}
//		return;
//	}

	updateModel(false);

//	if (updateNum > (dqnOption.explorePart * epochNum)) {
//		return;
//	}
//	float newEpsilon = (dqnOption.exploreBegin - dqnOption.exploreEnd) * (epochNum * dqnOption.explorePart - updateNum) / (epochNum * dqnOption.explorePart) + dqnOption.exploreEnd;
//	policy.updateEpsilon(newEpsilon);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void NoisyDoubleDqn<NetType, EnvType, PolicyType, OptimizerType>::test(const int epochNum, bool render) {
	tModel.eval();
	tester.testPlain();
	tModel.train();
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void NoisyDoubleDqn<NetType, EnvType, PolicyType, OptimizerType>::save() {
	if (!dqnOption.saveModel) {
		return;
	}

	AlgUtils::SaveModel(bModel, optimizer, dqnOption.savePathPrefix, logger);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void NoisyDoubleDqn<NetType, EnvType, PolicyType, OptimizerType>::load() {
	if (!dqnOption.loadModel) {
		return;
	}

	AlgUtils::LoadModel(bModel, optimizer, dqnOption.loadOptimizer, dqnOption.loadPathPrefix, logger);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void NoisyDoubleDqn<NetType, EnvType, PolicyType, OptimizerType>::saveByReward(float reward) {
	if (!dqnOption.saveModel) {
		return;
	}

	std::string path = dqnOption.savePathPrefix + "_" + std::to_string(reward);
	AlgUtils::SaveModel(tModel, optimizer, path, logger);
}

#endif /* INC_ALG_NOISYDOUBLEDQN_HPP_ */
