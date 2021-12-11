/*
 * a3cgradshared.hpp
 *
 *  Created on: Dec 10, 2021
 *      Author: zf
 */

#ifndef INC_ALG_A3CGRADSHARED_HPP_
#define INC_ALG_A3CGRADSHARED_HPP_



#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <tensorboard_logger.h>

#include <memory>
#include <vector>
#include <mutex>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/a2cstatestore.h"
#include "gymtest/utils/stats.h"
#include "gymtest/utils/lossstats.h"
#include "gymtest/utils/a2cnstore.h"
#include "alg/dqnoption.h"

//#include "a3c/a3ctcpclienthanle.hpp"
#include "a3c/a3cgradque.h"

#include "gymtest/utils/inputnorm.h"

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
class A3CGradShared {
private:
	NetType& bModel;
//	NetType& tModel;
	EnvType& env;
//	EnvType& testEnv;
	PolicyType& policy;
//	A3CGradQueue& q;
	OptimizerType& optimizer;
	std::mutex& updateMutex;
	const torch::Device deviceType;
	const at::IntArrayRef inputShape;

	const DqnOption dqnOption;
	TensorBoardLogger tLogger;

	const int batchSize;
	int offset = 0;

	uint32_t updateNum = 0;
	const int updateTargetGap; //TODO

	A2CNStorage rollout;
	const uint32_t maxStep;
	uint32_t qIndex = 0;

	const float entropyCoef;

	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("a3cgradshared");
	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);


	InputNorm rewardNorm;

	float maxAveReward;
	float maxSumReward;

	void save();
	void saveByReward(float reward);


public:
	const float gamma;
	const int testEp = 16;

	A3CGradShared(NetType& behaviorModel, EnvType& iEnv, PolicyType& iPolicy, OptimizerType& iOpt,
			std::mutex& iMutex,
			int stepSize,
			DqnOption option);
	~A3CGradShared() = default;
	A3CGradShared(const A3CGradShared& ) = delete;

	void train(const int epNum);

//	void test(const int batchSize, const int epochNum);
	void load();
};

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>

A3CGradShared<NetType, EnvType, PolicyType, OptimizerType>::A3CGradShared(
		NetType& behaviorModel, EnvType& iEnv, PolicyType& iPolicy, OptimizerType& iOpt,
		std::mutex& iMutex,
		int stepSize, const DqnOption iOption):
	bModel(behaviorModel),
//	tModel(trainModel),
	env(iEnv),
//	testEnv(tEnv),
	policy(iPolicy),
//	q(iQ),
	optimizer(iOpt),
	updateMutex(iMutex),
	dqnOption(iOption),
	deviceType(iOption.deviceType),
	inputShape(iOption.inputShape),
	gamma(iOption.gamma),
	tLogger(iOption.tensorboardLogPath.c_str()),
	batchSize(iOption.batchSize),
	updateTargetGap(iOption.targetUpdate),
	rollout(iOption.deviceType, stepSize),
	maxStep(stepSize),
	entropyCoef(iOption.entropyCoef),
	rewardNorm(iOption.deviceType),
	maxAveReward(iOption.saveThreshold),
	maxSumReward(iOption.sumSaveThreshold)
	{
	offset = 1;
	for (int i = 1; i < inputShape.size(); i ++) {
		offset *= inputShape[i];
	}
}

//template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
//void A3CGradShared<NetType, EnvType, PolicyType, OptimizerType>::test(const int batchSize, const int epochNum) {
//	LOG4CXX_INFO(logger, "To test " << epochNum << "episodes");
//	if (!dqnOption.toTest) {
//		return;
//	}
//
//	int epCount = 0;
//	int roundCount = 0;
//	std::vector<float> statRewards(batchSize, 0);
//	std::vector<float> statLens(batchSize, 0);
//	std::vector<float> sumRewards(batchSize, 0);
//	std::vector<float> sumLens(batchSize, 0);
//	std::vector<int> liveCounts(batchSize, 0);
//	std::vector<int> noReward(batchSize, 0);
//	std::vector<int> randomStep(batchSize, 0);
//
//	torch::NoGradGuard guard;
//	std::vector<float> states = testEnv.reset();
//	while (epCount < epochNum) {
////		torch::Tensor stateTensor = torch::from_blob(states.data(), inputShape).to(deviceType);
//		torch::Tensor stateTensor = torch::from_blob(states.data(), inputShape).div(dqnOption.inputScale).to(deviceType);
//
//		std::vector<torch::Tensor> rc = bModel.forward(stateTensor);
//		auto actionOutput = rc[0]; //TODO: detach?
//		auto valueOutput = rc[1];
//		auto actionProbs = torch::softmax(actionOutput, -1);
//		//TODO: To replace by getActions
//
//		std::vector<int64_t> actions = policy.getTestActions(actionProbs);
//
//		auto stepResult = testEnv.step(actions, true);
//		auto nextStateVec = std::get<0>(stepResult);
//		auto rewardVec = std::get<1>(stepResult);
//		auto doneVec = std::get<2>(stepResult);
//
//		Stats::UpdateReward(statRewards, rewardVec);
//		Stats::UpdateLen(statLens);
//
//		for (int i = 0; i < batchSize; i ++) {
//			if (doneVec[i]) {
//				LOG4CXX_DEBUG(logger, "testEnv " << i << "done");
////				auto resetResult = env.reset(i);
//				//udpate nextstatevec, target mask
////				std::copy(resetResult.begin(), resetResult.end(), nextStateVec.begin() + (offset * i));
//				epCount ++;
//
//				sumRewards[i] += statRewards[i];
//				sumLens[i] += statLens[i];
//
////				testStater.update(statLens[i], statRewards[i]);
//				tLogger.add_scalar("test/len", epCount, statLens[i]);
//				tLogger.add_scalar("test/reward", epCount, statRewards[i]);
//				LOG4CXX_INFO(logger, "test -----------> "<< i << " " << statLens[i] << ", " << statRewards[i]);
//				statLens[i] = 0;
//				statRewards[i] = 0;
////				stater.printCurStat();
//
//				liveCounts[i] ++;
//				if (liveCounts[i] >= dqnOption.donePerEp) {
//					roundCount ++;
//					LOG4CXX_INFO(logger, "Wrapper episode " << i << " ----------------------------> " << sumRewards[i]);
//					tLogger.add_scalar("test/sumLen", roundCount, sumLens[i]);
//					tLogger.add_scalar("test/sumReward", roundCount, sumRewards[i]);
////					sumStater.update(sumLens[i], sumRewards[i]);
//					liveCounts[i] = 0;
//					sumRewards[i] = 0;
//					sumLens[i] = 0;
//				}
//
//			}
//
//			if (dqnOption.randomHang) {
//			//Good action should have reward
//				if (rewardVec[i] < dqnOption.hangRewardTh) { //for float compare
//					noReward[i] ++;
//				}
//			}
//		}
//		states = nextStateVec;
//	}
//}


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>

void A3CGradShared<NetType, EnvType, PolicyType, OptimizerType>::train(const int epNum) {
	LOG4CXX_INFO(logger, "----------------------> batch");
	load();

	int step = 0;
	int epCount = 0;
//	int updateNum = 0;
	int roundNum = 0;

//	std::vector<int64_t> batchInputShape(1 + inputShape.size(), 0);
//	batchInputShape[0] = maxStep;
//	for (int i = 1; i < batchInputShape.size(); i ++) {
//		batchInputShape[i] = inputShape[i - 1];
//	}
	std::vector<int64_t> batchInputShape;
	if (dqnOption.isAtari) {
		batchInputShape.push_back(maxStep * batchSize);
		for (int i = 1; i < inputShape.size(); i ++) {
			batchInputShape.push_back(inputShape[i]);
		}
	} else {
//		batchInputShape = std::vector<int64_t>(1 + inputShape.size(), 0);
		batchInputShape.push_back(maxStep);
		for (int i = 0; i < inputShape.size(); i ++) {
			batchInputShape.push_back(inputShape[i]);
		}
	}

	std::vector<float> statRewards(batchSize, 0);
	std::vector<float> statLens(batchSize, 0);
	std::vector<int> liveCounts(batchSize, 0);
	std::vector<float> sumRewards(batchSize, 0);
	std::vector<float> sumLens(batchSize, 0);
	std::vector<float> clipRewards(batchSize, 0);
	std::vector<float> clipSumRewards(batchSize, 0);
	std::vector<float> idleStep(batchSize, 0);


	std::vector<float> stateVec = env.reset();
	while (updateNum < epNum) {
		std::vector<std::vector<float>> statesVec;
		std::vector<std::vector<float>> rewardsVec;
		std::vector<std::vector<float>> donesVec;
		std::vector<std::vector<long>> actionsVec;

		torch::Tensor stateTensor;
		torch::Tensor returnTensor;
		torch::Tensor actionTensor;

//		bModel.eval();
		{
			torch::NoGradGuard guard;

			for (step = 0; step < maxStep; step ++) {
				updateNum ++;

				torch::Tensor stateTensor = torch::from_blob(stateVec.data(), inputShape).div(dqnOption.inputScale).to(deviceType);
//				LOG4CXX_INFO(logger, "stateTensor in step: " << stateTensor.max() << stateTensor.mean());
				std::vector<torch::Tensor> rc = bModel.forward(stateTensor);
				auto actionProbs =  torch::softmax(rc[0], -1);
				std::vector<int64_t> actions = policy.getActions(actionProbs);
//				LOG4CXX_INFO(logger, "actions: " << actions);
//				LOG4CXX_INFO(logger, "state: " << stateTensor);


				auto stepResult = env.step(actions, false);
				auto nextStateVec = std::get<0>(stepResult);
				auto rewardVec = std::get<1>(stepResult);
				auto doneVec = std::get<2>(stepResult);

				Stats::UpdateReward(statRewards, rewardVec);
				Stats::UpdateLen(statLens);
				if (dqnOption.clipRewardStat) {
					Stats::UpdateReward(clipRewards, rewardVec, true, dqnOption.rewardMin, dqnOption.rewardMax);
				}

//				LOG4CXX_INFO(logger, "rewardVec" << step << ": " << rewardVec);
				std::vector<float> doneMaskVec(doneVec.size(), 1);
				for (int i = 0; i < doneVec.size(); i ++) {
//					LOG4CXX_INFO(logger, "dones: " << i << ": " << doneVec[i]);
					if (doneVec[i]) {
						doneMaskVec[i] = 0;
						epCount ++;

						sumRewards[i] += statRewards[i];
						sumLens[i] += statLens[i];

						tLogger.add_scalar("train/len", updateNum, statLens[i]);
						tLogger.add_scalar("train/reward", updateNum, statRewards[i]);
						LOG4CXX_INFO(logger, "" << updateNum << ": " << statLens[i] << ", " << statRewards[i]);
						statLens[i] = 0;
						statRewards[i] = 0;

						if (dqnOption.multiLifes) {
							liveCounts[i] ++;
							if (liveCounts[i] >= dqnOption.donePerEp) {
								roundNum ++;
								LOG4CXX_INFO(logger, "Wrapper episode " << i << " ----------------------------> " << sumRewards[i]);
								tLogger.add_scalar("train/epLen", roundNum, sumLens[i]);
								tLogger.add_scalar("train/epReward", roundNum, sumRewards[i]);

								liveCounts[i] = 0;
								sumRewards[i] = 0;
								sumLens[i] = 0;
								clipSumRewards[i] = 0;
							}
						}
					}
				}


				statesVec.push_back(stateVec);
				rewardsVec.push_back(rewardVec);
				donesVec.push_back(doneMaskVec);
				actionsVec.push_back(actions);

				stateVec = nextStateVec;
			}


		//TODO: Can make use of grad summary property?
/////////////////////////////////////////// Prepare
			torch::Tensor lastStateTensor = torch::from_blob(stateVec.data(), inputShape).div(dqnOption.inputScale).to(deviceType);
//			LOG4CXX_INFO(logger, "stateTensor in last: " << lastStateTensor.max() << lastStateTensor.mean());

			auto rc = bModel.forward(lastStateTensor);
			auto lastValueTensor = rc[1].squeeze(-1);
//			LOG4CXX_INFO(logger, "lastValue: " << lastValueTensor);

			auto stateData = EnvUtils::FlattenVector(statesVec);
			stateTensor = torch::from_blob(stateData.data(), batchInputShape).div(dqnOption.inputScale).to(deviceType);
//			LOG4CXX_INFO(logger, "stateTensor in batch: " << stateTensor.max() << stateTensor.mean());

			auto actionData = EnvUtils::FlattenVector(actionsVec);
			actionTensor = torch::from_blob(actionData.data(), {maxStep, batchSize, 1}, longOpt).to(deviceType);
			auto maskData = EnvUtils::FlattenVector(donesVec);
			torch::Tensor maskTensor = torch::from_blob(maskData.data(), {maxStep, batchSize}).to(deviceType);
			auto rewardData = EnvUtils::FlattenVector(rewardsVec);
			torch::Tensor rewardTensor = torch::from_blob(rewardData.data(), {maxStep, batchSize}).to(deviceType);
			rewardTensor = rewardTensor.div(dqnOption.rewardScale).clamp(dqnOption.rewardMin, dqnOption.rewardMax);


			returnTensor = torch::zeros({maxStep, batchSize}).to(deviceType);
			torch::Tensor qTensor = lastValueTensor;
//			LOG4CXX_INFO(logger, "qTensor begin: " << qTensor);
			for (int i = maxStep - 1; i >= 0; i --) {
				qTensor = qTensor * maskTensor[i] * gamma + rewardTensor[i];
				returnTensor[i].copy_(qTensor);
			}
			if (dqnOption.normReward) {
				returnTensor = (returnTensor - returnTensor.mean()) / (returnTensor.std() + 1e-7);
			}
			returnTensor = returnTensor.detach();
		}
///////////////////////////////////////////// Calculation
		float lossV, aLossV, vLossV, entropyV;
		{
			std::unique_lock<std::mutex> lock(updateMutex);

//			bModel.zero_grad();
//			bModel.train();
			optimizer.zero_grad();

			auto output = bModel.forward(stateTensor);
			auto actionOutputTensor = output[0].squeeze(-1).view({maxStep, batchSize, -1}); //{maxstep * batch, actionNum, 1} -> {maxstep * batch, actionNum} -> {maxstep, batch, actionNum}
			auto valueTensor = output[1].squeeze(-1).view({maxStep, batchSize});
//			LOG4CXX_INFO(logger, "compare value " << tmpValue);
//			LOG4CXX_INFO(logger, "batch value: " << valueTensor);
			auto actionProbTensor = torch::softmax(actionOutputTensor, -1); //{maxstep, batch, actionNum}
			auto actionLogTensor = torch::log_softmax(actionOutputTensor, -1); //{maxStep, batch, actionNum}

			torch::Tensor entropyLoss = - (actionProbTensor * actionLogTensor).sum(-1).mean();

			auto advTensor = returnTensor - valueTensor;
//			LOG4CXX_INFO(logger, "returnTensor: " << returnTensor);
//			LOG4CXX_INFO(logger, "valueTensor: " << valueTensor);
			torch::Tensor valueLoss = torch::nn::functional::mse_loss(valueTensor, returnTensor);

//			LOG4CXX_INFO(logger, "actionLogTensor: " << actionLogTensor.sizes());
//			LOG4CXX_INFO(logger, "actionTensor: " << actionTensor.sizes());
			auto actPiTensor = actionLogTensor.gather(-1, actionTensor).squeeze(-1);
			torch::Tensor actLoss = - (actPiTensor * advTensor.detach()).mean();

			torch::Tensor loss = dqnOption.valueCoef * valueLoss + actLoss - dqnOption.entropyCoef * entropyLoss;

			lossV = loss.item<float>();
			aLossV = actLoss.item<float>();
			vLossV = valueLoss.item<float>();
			entropyV = entropyLoss.item<float>();

			loss.backward();
			torch::nn::utils::clip_grad_norm_(bModel.parameters(), dqnOption.maxGradNormClip);
			optimizer.step();
		}
		LOG4CXX_INFO(logger, "loss" << updateNum << ": " << lossV
			<< ", " << vLossV << ", " << aLossV << ", " << entropyV);
		if ((updateNum % dqnOption.logInterval) == 0) {
			tLogger.add_scalar("loss/loss", updateNum, lossV);
			tLogger.add_scalar("loss/vLoss", updateNum, vLossV);
			tLogger.add_scalar("loss/aLoss", updateNum, aLossV);
			tLogger.add_scalar("loss/entropy", updateNum, entropyV);
		}
	}

	save();
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>

void A3CGradShared<NetType, EnvType, PolicyType, OptimizerType>::save() {
	if (!dqnOption.saveModel) {
		return;
	}

	std::string modelPath = dqnOption.savePathPrefix + "_model.pt";
	torch::serialize::OutputArchive outputArchive;
	bModel.save(outputArchive);
	outputArchive.save_to(modelPath);
	LOG4CXX_INFO(logger, "Save reward model into " << modelPath);

	std::string optPath = dqnOption.savePathPrefix + "_optimizer.pt";
	torch::serialize::OutputArchive optimizerArchive;
//	optimizer.save(optimizerArchive);
	optimizerArchive.save_to(optPath);
	LOG4CXX_INFO(logger, "Save reward optimizer into " << optPath);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>

void A3CGradShared<NetType, EnvType, PolicyType, OptimizerType>::saveByReward(float reward) {
	if (!dqnOption.saveModel) {
		return;
	}

	std::string modelPath = dqnOption.savePathPrefix + "_" + std::to_string(reward) + "_model.pt";
	torch::serialize::OutputArchive outputArchive;
	bModel.save(outputArchive);
	outputArchive.save_to(modelPath);
	LOG4CXX_INFO(logger, "Save model into " << modelPath);

	std::string optPath = dqnOption.savePathPrefix + "_" + std::to_string(reward) + "_optimizer.pt";
	torch::serialize::OutputArchive optimizerArchive;
//	optimizer.save(optimizerArchive);
	optimizerArchive.save_to(optPath);
	LOG4CXX_INFO(logger, "Save optimizer into " << optPath);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A3CGradShared<NetType, EnvType, PolicyType, OptimizerType>::load() {
	if (!dqnOption.loadModel) {
		return;
	}

	std::string modelPath = dqnOption.loadPathPrefix + "_model.pt";
	torch::serialize::InputArchive inChive;
	inChive.load_from(modelPath);
	bModel.load(inChive);
	LOG4CXX_INFO(logger, "Load model from " << modelPath);

//	updateTarget();

	if (dqnOption.loadOptimizer) {
		std::string optPath = dqnOption.loadPathPrefix + "_optimizer.pt";
		torch::serialize::InputArchive opInChive;
		opInChive.load_from(optPath);
//		optimizer.load(opInChive);
		LOG4CXX_INFO(logger, "Load optimizer from " << optPath);
	}

}



#endif /* INC_ALG_A3CGRADSHARED_HPP_ */
