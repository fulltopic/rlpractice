/*
 * catdqn.hpp
 *
 *  Created on: Oct 21, 2021
 *      Author: zf
 */

#ifndef INC_ALG_CATDQN_HPP_
#define INC_ALG_CATDQN_HPP_



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
#include "gymtest/utils/lossstats.h"
#include "dqnoption.h"

#include "utils/utils.hpp"
#include "utils/algtester.hpp"
#include "utils/replaybuffer.h"

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
class CategoricalDqn {
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

	torch::Tensor valueItems;
	torch::Tensor offset;
	float deltaZ = 0;

	//log
	const torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("dqn");
	TensorBoardLogger tLogger;

	AlgTester<NetType, EnvType, PolicyType> tester;


	ReplayBuffer buffer; //buffer has to be defined after dqnOption so ReplayBuffer can get all parameters of dqnOption.

	void updateModel(bool force = false);
	void updateStep(const float epochNum);

	void load();
	void save();
	void saveTModel(float reward);
public:
	CategoricalDqn(NetType& iModel, NetType& iTModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, DqnOption iOption);
	~CategoricalDqn() = default;
	CategoricalDqn(const CategoricalDqn&) = delete;

	void train(const int epochNum);
	void test(const int epochNum, bool render = false, bool toLoad = true);
};


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
CategoricalDqn<NetType, EnvType, PolicyType, OptimizerType>::CategoricalDqn(NetType& iModel, NetType& iTModel,
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
	tLogger(iOption.tensorboardLogPath.c_str()),
	tester(iTModel, tEnv, iPolicy, iOption, tLogger)
{
//	maxTestReward = iOption.saveThreshold;

	valueItems = torch::linspace(dqnOption.vMin, dqnOption.vMax, dqnOption.atomNum).to(dqnOption.deviceType);
	offset = (torch::linspace(0, dqnOption.batchSize - 1, dqnOption.batchSize) * dqnOption.atomNum).unsqueeze(-1).to(torch::kLong).to(dqnOption.deviceType); //{batchSize * atomNum, 1}
	deltaZ = (dqnOption.vMax - dqnOption.vMin) / ((float)(dqnOption.atomNum - 1));
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void CategoricalDqn<NetType, EnvType, PolicyType, OptimizerType>::train(const int epochNum) {
	load();
	updateModel(true); //model assignment
	tModel.eval();

	std::vector<float> stateVec = env.reset();

	//only one env
	std::vector<float> statRewards(dqnOption.envNum, 0);
	std::vector<float> statLens(dqnOption.envNum, 0);
	std::vector<float> statSumRewards(dqnOption.envNum, 0);
	std::vector<float> statSumLens(dqnOption.envNum, 0);
	std::vector<int> livePerEp(dqnOption.envNum, 0);
	int epNum = 0;

	while (updateNum < epochNum) {
		for (int k = 0; k < dqnOption.envStep; k ++) {
			updateNum ++;
			//Run step
			torch::Tensor cpuinputTensor = torch::from_blob(stateVec.data(), inputShape);
			torch::Tensor inputTensor = cpuinputTensor.to(deviceType).div(dqnOption.inputScale);

			torch::Tensor outputTensor = bModel.forward(inputTensor);
			outputTensor = outputTensor.view({1, dqnOption.outputNum, dqnOption.atomNum});
			outputTensor = torch::softmax(outputTensor, -1).squeeze(0);
//			LOG4CXX_INFO(logger, "predict dist: " << outputTensor);
			outputTensor = outputTensor * valueItems; //ValueItems expanded properly
			outputTensor = outputTensor.sum(-1, false);
//			LOG4CXX_INFO(logger, "input state: " << inputTensor);
//			LOG4CXX_INFO(logger, "predict values: " << outputTensor);
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

			torch::Tensor nextInputTensor = torch::from_blob(nextInputVec.data(), inputShape); //.div(dqnOption.inputScale);
//			float reward = rewardVec[0]; //Would be clipped in atom operation
			buffer.add(cpuinputTensor, nextInputTensor, actions[0], rewardVec[0], doneMask);

			stateVec = nextInputVec;
			updateStep(epochNum);
		} //End of envStep

		//Learning
		if (updateNum < dqnOption.startStep) {
			continue;
		}

		float updateLoss = 0;
		float updateQs = 0;
		for (int k = 0; k < dqnOption.epochPerUpdate; k ++) {
			//sample
			torch::Tensor sampleIndice = buffer.getSampleIndex(dqnOption.batchSize);
			torch::Tensor curStateTensor = buffer.states.index_select(0, sampleIndice).to(deviceType);
			torch::Tensor actionTensor = buffer.actions.index_select(0, sampleIndice).to(deviceType);
			torch::Tensor rewardTensor = buffer.rewards.index_select(0, sampleIndice).to(deviceType); //{batchSize, 1}
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

			//Calculate target
			torch::Tensor targetDist;
			LOG4CXX_DEBUG(logger, "targetQ before " << targetDist);
			{
				torch::NoGradGuard guard;

				torch::Tensor nextOutput = tModel.forward(nextStateTensor).detach();
				nextOutput = nextOutput.view({dqnOption.batchSize, dqnOption.outputNum, dqnOption.atomNum});
				torch::Tensor nextProbs = torch::softmax(nextOutput, -1); //{batch, action, atom}

				torch::Tensor nextQs = nextProbs * valueItems; // valueItems expands
				nextQs = nextQs.sum(-1, false); //{batchSize, actionNum}
				auto nextMaxOutput = nextQs.max(-1, true); //{batchSize, actionNum}
				torch::Tensor nextMaxQs = std::get<0>(nextMaxOutput); //{batchSize, 1}
				torch::Tensor nextMaxActions = std::get<1>(nextMaxOutput); //{batchSize, 1}
				nextMaxActions = nextMaxActions.unsqueeze(1).expand({dqnOption.batchSize, 1, dqnOption.atomNum});
				torch::Tensor nextDist = nextProbs.gather(1, nextMaxActions).squeeze(1); //{batchSize, atomNum}

				torch::Tensor shiftValues = rewardTensor + dqnOption.gamma * doneMaskTensor * valueItems; //{batchSize, atomNum}
				shiftValues = shiftValues.clamp(dqnOption.vMin, dqnOption.vMax);
				torch::Tensor shiftIndex = (shiftValues - dqnOption.vMin) / deltaZ; //{batchSize, atomNum}
				torch::Tensor l = shiftIndex.floor();
				torch::Tensor u = shiftIndex.ceil();
				torch::Tensor lIndice = l.to(torch::kLong);
				torch::Tensor uIndice = u.to(torch::kLong);
				torch::Tensor lDelta = u - shiftIndex;
				torch::Tensor uDelta = shiftIndex - l;
				torch::Tensor eqIndice = lIndice.eq(uIndice).to(torch::kFloat); //bool to float
				lDelta.add_(eqIndice); //shiftIndex % deltaZ == 0

				//Prepare for index_add
				torch::Tensor lDist = (nextDist * lDelta).view({dqnOption.batchSize * dqnOption.atomNum});  //{batchSize, atomNum}
				torch::Tensor uDist = (nextDist * uDelta).view({dqnOption.batchSize * dqnOption.atomNum});

				targetDist = torch::zeros({dqnOption.batchSize * dqnOption.atomNum}).to(dqnOption.deviceType);

				lIndice = (lIndice + offset).view({dqnOption.batchSize * dqnOption.atomNum}); //offset expanded
				uIndice = (uIndice + offset).view({dqnOption.batchSize * dqnOption.atomNum});
				targetDist.index_add_(0, lIndice, lDist);
				targetDist.index_add_(0, uIndice, uDist);

				targetDist = targetDist.view({dqnOption.batchSize, dqnOption.atomNum});
//				LOG4CXX_INFO(logger, "targetDist: " << targetDist);
//				LOG4CXX_INFO(logger, "new dist: " << targetDist.sum(-1));
			}

			//Calculate current Q
			torch::Tensor curOutput = bModel.forward(curStateTensor);
			curOutput = curOutput.view({dqnOption.batchSize, dqnOption.outputNum, dqnOption.atomNum});
			actionTensor = actionTensor.unsqueeze(1).expand({dqnOption.batchSize, 1, dqnOption.atomNum});
			torch::Tensor curDist = curOutput.gather(1, actionTensor).squeeze(1); //{batchSize, atomNum}
			torch::Tensor curLogDist = torch::log_softmax(curDist, -1);
			LOG4CXX_DEBUG(logger, "curOutput: " << curOutput);
//			LOG4CXX_INFO(logger, "curLogDist: " << curLogDist);

			//Update
			auto loss = -(targetDist * curLogDist).sum(-1).mean();

			optimizer.zero_grad();
			loss.backward();
			torch::nn::utils::clip_grad_norm_(bModel.parameters(), dqnOption.maxGradNormClip);
			optimizer.step();

			updateLoss += loss.item<float>();
			updateQs += (curDist.softmax(-1) * valueItems).sum(-1, false).mean().item<float>();
		}

		if ((updateNum % dqnOption.logInterval) == 0) {
//			auto lossValue = loss.item<float>();
//			auto qs = (curDist.softmax(-1) * valueItems).sum(-1, false).mean();
//			auto qsValue = qs.item<float>();

			tLogger.add_scalar("loss/entLoss", updateNum, updateLoss / (float)dqnOption.epochPerUpdate);
			tLogger.add_scalar("loss/qs", updateNum, updateQs / (float)dqnOption.epochPerUpdate);
		}

		//TEST
		if (dqnOption.toTest) {
			if ((updateNum % dqnOption.testGapEp) == 0) {
				test(dqnOption.testEp, false, false);
			}
		}
	}

	save();
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void CategoricalDqn<NetType, EnvType, PolicyType, OptimizerType>::updateModel(bool force) {
	if (!force) {
		if ((updateNum % dqnOption.targetUpdateStep) != 0) {
			return;
		}
	}

	torch::NoGradGuard guard;

	AlgUtils::SyncNet(bModel, tModel, dqnOption.tau);
	LOG4CXX_DEBUG(logger, "----------------- >target network synched");
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void CategoricalDqn<NetType, EnvType, PolicyType, OptimizerType>::updateStep(const float epochNum) {
	updateModel(false);

	if (updateNum > (dqnOption.explorePart * epochNum)) {
		return;
	}
	float newEpsilon = (dqnOption.exploreBegin - dqnOption.exploreEnd) * (epochNum * dqnOption.explorePart - updateNum) / (epochNum * dqnOption.explorePart) + dqnOption.exploreEnd;
	policy.updateEpsilon(newEpsilon);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void CategoricalDqn<NetType, EnvType, PolicyType, OptimizerType>::test(const int epochNum, bool render, bool toLoad) {
	if (toLoad) {
		load();
		updateModel();
	}

	tester.testCategory();
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void CategoricalDqn<NetType, EnvType, PolicyType, OptimizerType>::save() {
	if (!dqnOption.saveModel) {
		return;
	}

	AlgUtils::SaveModel(bModel, optimizer, dqnOption.savePathPrefix, logger);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void CategoricalDqn<NetType, EnvType, PolicyType, OptimizerType>::saveTModel(float reward) {
	if (!dqnOption.saveModel) {
		return;
	}

	std::string modelPath = dqnOption.savePathPrefix + "_" + std::to_string(reward);
	AlgUtils::SaveModel(tModel, optimizer, modelPath, logger);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void CategoricalDqn<NetType, EnvType, PolicyType, OptimizerType>::load() {
	if (!dqnOption.loadModel) {
		return;
	}

	AlgUtils::LoadModel(bModel, optimizer, dqnOption.loadOptimizer, dqnOption.loadPathPrefix, logger);

}


#endif /* INC_ALG_CATDQN_HPP_ */
