/*
 * dqnzip.hpp
 *
 *  Created on: Oct 9, 2021
 *      Author: zf
 */

#ifndef INC_ALG_DQNZIP_HPP_
#define INC_ALG_DQNZIP_HPP_


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
#include "dqnoption.h"

#include "utils/utils.hpp"
#include "utils/replaybuffer.h"

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
class DqnZip {
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
	float maxTestReward = -1000;

	//Gym server failed to reset by reset()
	std::vector<float> testRewards;
	std::vector<float> testLens;
	std::vector<float> testEpRewards;
	std::vector<float> testEpLens;
	std::vector<int> testLivePerEp;

	int totalTestLive = 0;
	int totalTestEp = 0;

	const torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("dqn");

	TensorBoardLogger tLogger;

	ReplayBuffer buffer; //buffer has to be defined after dqnOption so ReplayBuffer can get all parameters of dqnOption.

	void updateModel(bool force = false);
	void updateStep(const float epochNum);

	void load();
	void save();
	void saveTModel(float reward);
public:
	DqnZip(NetType& iModel, NetType& iTModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, DqnOption iOption);
	~DqnZip() = default;
	DqnZip(const DqnZip&) = delete;

	void train(const int epochNum);
	void test(const int epochNum, bool render = false, bool toLoad = true);
};


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
DqnZip<NetType, EnvType, PolicyType, OptimizerType>::DqnZip(NetType& iModel, NetType& iTModel,
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
	buffer(iOption.rbCap, iOption.inputShape),
	tLogger(iOption.tensorboardLogPath.c_str())
{
	maxTestReward = iOption.saveThreshold;

	testRewards = std::vector<float>(dqnOption.testBatch, 0);
	testLens = std::vector<float>(dqnOption.testBatch, 0);
	testEpRewards = std::vector<float>(dqnOption.testBatch, 0);
	testEpLens = std::vector<float>(dqnOption.testBatch, 0);
	testLivePerEp = std::vector<int>(dqnOption.testBatch, 0);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void DqnZip<NetType, EnvType, PolicyType, OptimizerType>::train(const int epochNum) {
	load();
	updateModel(true); //model assignment
	tModel.eval();

	std::vector<float> stateVec = env.reset();

	//only one env
	std::vector<float> statRewards(dqnOption.envNum, 0);
	std::vector<float> statLens(dqnOption.envNum, 0);

	while (updateNum < epochNum) {
		for (int k = 0; k < dqnOption.envStep; k ++) {
			updateNum ++;
			//Run step
			torch::Tensor cpuinputTensor = torch::from_blob(stateVec.data(), inputShape);
			torch::Tensor inputTensor = cpuinputTensor.to(deviceType).div(dqnOption.inputScale);

			torch::Tensor outputTensor = bModel.forward(inputTensor);
			std::vector<int64_t> actions = policy.getActions(outputTensor);

			auto stepResult = env.step(actions);
			auto nextInputVec = std::get<0>(stepResult);
			auto rewardVec = std::get<1>(stepResult);
			auto doneVec = std::get<2>(stepResult);

			Stats::UpdateReward(statRewards, rewardVec);
			Stats::UpdateLen(statLens);
			float doneMask = 1;
			if (doneVec[0]) {
				tLogger.add_scalar("train/reward", updateNum, statRewards[0]);
				tLogger.add_scalar("train/len", updateNum, statLens[0]);
				LOG4CXX_INFO(logger, "" << policy.getEpsilon() << "--" << updateNum << ": " << statLens[0] << ", " << statRewards[0]);
				doneMask = 0;
				statRewards[0] = 0;
				statLens[0] = 0;
			}

			torch::Tensor nextInputTensor = torch::from_blob(nextInputVec.data(), inputShape); //.div(dqnOption.inputScale);
			float reward = std::max(std::min((rewardVec[0] * dqnOption.rewardScale), dqnOption.rewardMax), dqnOption.rewardMin);
			buffer.add(cpuinputTensor, nextInputTensor, actions[0], reward, doneMask);

			//Update
			stateVec = nextInputVec;
			updateStep(epochNum);
		} //End of envStep

		//TEST
		if (dqnOption.toTest) {
			if ((updateNum % dqnOption.testGapEp) == 0) {
				test(dqnOption.testEp, false, false);
			}
		}

		//Learning
		if (updateNum < dqnOption.startStep) {
			continue;
		}

		torch::Tensor sampleIndice = buffer.getSampleIndex(dqnOption.batchSize);
		torch::Tensor curStateTensor = buffer.states.index_select(0, sampleIndice).to(deviceType);
		torch::Tensor actionTensor = buffer.actions.index_select(0, sampleIndice).to(deviceType);
		torch::Tensor rewardTensor = buffer.rewards.index_select(0, sampleIndice).to(deviceType);
		torch::Tensor doneMaskTensor = buffer.donesMask.index_select(0, sampleIndice).to(deviceType);
		LOG4CXX_DEBUG(logger, "sampleIndex before: " << sampleIndice);
		auto nextSampleIndice = (sampleIndice + 1) % dqnOption.rbCap;
		torch::Tensor nextStateTensor = buffer.states.index_select(0, nextSampleIndice).to(deviceType);

		curStateTensor = curStateTensor.to(torch::kFloat).div(dqnOption.inputScale);
		nextStateTensor = nextStateTensor.to(torch::kFloat).div(dqnOption.inputScale);
		rewardTensor = rewardTensor.to(torch::kFloat);
		doneMaskTensor = doneMaskTensor.to(torch::kFloat);
		actionTensor = actionTensor.to(torch::kLong);

		LOG4CXX_DEBUG(logger, "sampleIndice after: " << sampleIndice);
		LOG4CXX_DEBUG(logger, "nextStateTensor: " << nextStateTensor);

		torch::Tensor targetQ;
		LOG4CXX_DEBUG(logger, "targetQ before " << targetQ);
		{
			torch::NoGradGuard guard; //replaced by tModel.eval()?
			torch::Tensor nextOutput = tModel.forward(nextStateTensor).detach();
			LOG4CXX_DEBUG(logger, "nextOutput: " << nextOutput);
			auto maxOutput = nextOutput.max(-1);
			torch::Tensor nextQ = std::get<0>(maxOutput); //pay attention to shape
			nextQ = nextQ.unsqueeze(1);
			LOG4CXX_DEBUG(logger, "nextQ: " << nextQ);
			LOG4CXX_DEBUG(logger, "rewardTensor: " << rewardTensor);
			LOG4CXX_DEBUG(logger, "doneMaskTensor: " << doneMaskTensor);

			targetQ = rewardTensor + dqnOption.gamma * nextQ * doneMaskTensor;
			LOG4CXX_DEBUG(logger, "targetQ: " << targetQ);
		}

		torch::Tensor curOutput = bModel.forward(curStateTensor);
		LOG4CXX_DEBUG(logger, "curOutput: " << curOutput);
		torch::Tensor curQ = curOutput.gather(-1, actionTensor); //shape of actionTensor and curQ
		LOG4CXX_DEBUG(logger, "curQ: " << curQ);

		auto loss = torch::nn::functional::smooth_l1_loss(curQ, targetQ);
		//TODO: Try mse.
//		auto loss = torch::nn::functional::mse_loss(curQ, targetQ);

		if ((updateNum % dqnOption.logInterval) == 0) {
			float lossValue = loss.item<float>();
			float qValue = curQ.mean().item<float>();
			tLogger.add_scalar("loss/loss", updateNum, lossValue);
			tLogger.add_scalar("loss/qValue", updateNum, qValue);
			tLogger.add_scalar("loss/epsilon", updateNum, policy.getEpsilon());
		}

		optimizer.zero_grad();
		loss.backward();
		torch::nn::utils::clip_grad_norm_(bModel.parameters(), dqnOption.maxGradNormClip);
		optimizer.step();
	}

	save();
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void DqnZip<NetType, EnvType, PolicyType, OptimizerType>::updateModel(bool force) {
	if (!force) {
		if ((updateNum % dqnOption.targetUpdateStep) != 0) {
			return;
		}
	}

	AlgUtils::SyncNet(bModel, tModel, dqnOption.tau);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void DqnZip<NetType, EnvType, PolicyType, OptimizerType>::updateStep(const float epochNum) {
	updateModel(false);

	if (updateNum > (dqnOption.explorePart * epochNum)) {
		return;
	}
	float newEpsilon = (dqnOption.exploreBegin - dqnOption.exploreEnd) * (epochNum * dqnOption.explorePart - updateNum) / (epochNum * dqnOption.explorePart) + dqnOption.exploreEnd;
	policy.updateEpsilon(newEpsilon);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void DqnZip<NetType, EnvType, PolicyType, OptimizerType>::test(const int epochNum, bool render, bool toLoad) {
	if (toLoad) {
		load();
		updateModel();
	}
	torch::NoGradGuard guard;


	std::vector<long> testShapeData;
	testShapeData.push_back(dqnOption.testBatch);
	for (int i = 1; i < inputShape.size(); i ++) {
		testShapeData.push_back(inputShape[i]);
	}
	at::IntArrayRef testInputShape(testShapeData);

	int epUpdate = 0;
	float totalLen = 0;
	float totalReward = 0;

	std::vector<float> stateVec = testEnv.reset();
	while (epUpdate < dqnOption.testEp) {
		torch::Tensor inputTensor = torch::from_blob(stateVec.data(), testInputShape).div(dqnOption.inputScale).to(deviceType);
		torch::Tensor outputTensor = tModel.forward(inputTensor);
		std::vector<int64_t> actions = policy.getTestActions(outputTensor);
		LOG4CXX_DEBUG(logger, "actions: " << actions);

		auto stepResult = testEnv.step(actions, render);
		auto nextInputVec = std::get<0>(stepResult);
		auto rewardVec = std::get<1>(stepResult);
		auto doneVec = std::get<2>(stepResult);
		LOG4CXX_DEBUG(logger, "rewardVec: " << rewardVec);

		Stats::UpdateReward(testRewards, rewardVec);
		Stats::UpdateLen(testLens);
		Stats::UpdateReward(testEpRewards, rewardVec);
		Stats::UpdateLen(testEpLens);

		for (int i = 0; i < dqnOption.testBatch; i ++) {
			if (doneVec[i]) {

				totalTestLive ++;
				LOG4CXX_INFO(logger, "ep" << totalTestLive << ": " << testRewards[i] << ", " << testLens[i]);
				tLogger.add_scalar("test/len", totalTestLive, testLens[i]);
				tLogger.add_scalar("test/reward", totalTestLive, testRewards[i]);
				testRewards[i] = 0;
				testLens[i] = 0;

//				epUpdate ++;
				testLivePerEp[i] ++;
				if (dqnOption.multiLifes) {
					if (testLivePerEp[i] == dqnOption.livePerEpisode) {
						epUpdate ++;
						totalTestEp ++;
						totalLen += testEpLens[i];
						totalReward += testEpRewards[i];

						tLogger.add_scalar("test/totalreward", totalTestEp, testEpRewards[i]);
						tLogger.add_scalar("test/totallen", totalTestEp, testEpLens[i]);

						testEpRewards[i] = 0;
						testEpLens[i] = 0;
						testLivePerEp[i] = 0;
					}
				} else {
					epUpdate ++;
				}
			}
		}

		stateVec = nextInputVec;
	}

//	float aveLen = totalLen / (float)dqnOption.testEp;
//	tLogger.add_scalar("test/ave_reward", updateNum, aveReward);
//	tLogger.add_scalar("test/ave_len", updateNum, aveLen);

	if (dqnOption.saveModel) {
		float aveReward = totalReward / (float)dqnOption.testEp;
		if (aveReward > maxTestReward) {
			maxTestReward = aveReward + dqnOption.saveStep;
			saveTModel(aveReward);
		}
	}
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void DqnZip<NetType, EnvType, PolicyType, OptimizerType>::save() {
	if (!dqnOption.saveModel) {
		return;
	}

	AlgUtils::SaveModel(bModel, optimizer, dqnOption.savePathPrefix, logger);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void DqnZip<NetType, EnvType, PolicyType, OptimizerType>::saveTModel(float reward) {
	if (!dqnOption.saveModel) {
		return;
	}

	std::string path = dqnOption.savePathPrefix + "_" + std::to_string(reward);
	AlgUtils::SaveModel(tModel, optimizer, path, logger);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void DqnZip<NetType, EnvType, PolicyType, OptimizerType>::load() {
	if (!dqnOption.loadModel) {
		return;
	}

	AlgUtils::LoadModel(bModel, optimizer, dqnOption.loadOptimizer, dqnOption.loadPathPrefix, logger);
}



#endif /* INC_ALG_DQNZIP_HPP_ */
