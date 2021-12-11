/*
 * a3ctest.hpp
 *
 *  Created on: Nov 20, 2021
 *      Author: zf
 */

#ifndef INC_ALG_A3CTEST_HPP_
#define INC_ALG_A3CTEST_HPP_




#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <tensorboard_logger.h>

#include <memory>
#include <vector>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/a2cstatestore.h"
#include "gymtest/utils/stats.h"
#include "gymtest/utils/lossstats.h"
#include "gymtest/utils/a2cnstore.h"
#include "alg/dqnoption.h"

#include "a3c/a3ctcpclienthanle.hpp"

#include "gymtest/utils/inputnorm.h"

template<typename NetType, typename OptimizerType, typename EnvType, typename PolicyType>
class A3CNStepTest {
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
	TensorBoardLogger tLogger;


	const int batchSize;
	int offset = 0;

	uint32_t updateNum = 0;
	const int updateTargetGap; //TODO

	A2CNStorage rollout;
	const uint32_t maxStep;
	uint32_t qIndex = 0;

	const float entropyCoef;

	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("a3ctest");
	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	torch::nn::HuberLoss huberLossComputer = torch::nn::HuberLoss();
	torch::nn::MSELoss mseLossComputer = torch::nn::MSELoss();

//	std::shared_ptr<A3CTCPClientHandle<NetType>> client;

	InputNorm rewardNorm;

	float maxAveReward;
	float maxSumReward;

	void save();
	void saveByReward(float reward);
	void syncFromTarget();

	void trainStep(const int epNum); //no batched
	void trainBatch(const int epNum); //batched
//	void testTrain(torch::Tensor input);

public:
	const float gamma;
	const int testEp = 16;

	A3CNStepTest(NetType& behaviorModel, NetType& targetModel, OptimizerType& opt, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, int stepSize, DqnOption option);
	~A3CNStepTest() = default;
	A3CNStepTest(const A3CNStepTest& ) = delete;

	void train(const int epNum, bool batched = false);

	void test(const int batchSize, const int epochNum);
	void load();
};

template<typename NetType, typename OptimizerType, typename EnvType, typename PolicyType>
A3CNStepTest<NetType, OptimizerType, EnvType, PolicyType>::A3CNStepTest(NetType& behaviorModel, NetType& targetModel, OptimizerType& opt, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy,
		int stepSize, const DqnOption iOption):
	bModel(behaviorModel),
	tModel(targetModel),
	env(iEnv),
	testEnv(tEnv),
	policy(iPolicy),
	optimizer(opt),
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
//	client(iClient)
{
	offset = 1;
	for (int i = 1; i < inputShape.size(); i ++) {
		offset *= inputShape[i];
	}
}

template<typename NetType, typename OptimizerType, typename EnvType, typename PolicyType>
void A3CNStepTest<NetType, OptimizerType, EnvType, PolicyType>::test(const int batchSize, const int epochNum) {
	LOG4CXX_INFO(logger, "To test " << epochNum << "episodes");
	if (!dqnOption.toTest) {
		return;
	}

	int epCount = 0;
	int roundCount = 0;
	std::vector<float> statRewards(batchSize, 0);
	std::vector<float> statLens(batchSize, 0);
	std::vector<float> sumRewards(batchSize, 0);
	std::vector<float> sumLens(batchSize, 0);
	std::vector<int> liveCounts(batchSize, 0);
	std::vector<int> noReward(batchSize, 0);
	std::vector<int> randomStep(batchSize, 0);

	torch::NoGradGuard guard;
	std::vector<float> states = testEnv.reset();
	while (epCount < epochNum) {
//		torch::Tensor stateTensor = torch::from_blob(states.data(), inputShape).to(deviceType);
		torch::Tensor stateTensor = torch::from_blob(states.data(), inputShape).div(dqnOption.inputScale).to(deviceType);

		std::vector<torch::Tensor> rc = bModel.forward(stateTensor);
		auto actionOutput = rc[0]; //TODO: detach?
		auto valueOutput = rc[1];
		auto actionProbs = torch::softmax(actionOutput, -1);
		//TODO: To replace by getActions

		std::vector<int64_t> actions = policy.getTestActions(actionProbs);
		if (dqnOption.randomHang) {
			for (int i = 0; i < batchSize; i ++) {
				if (noReward[i] >= dqnOption.hangNumTh) {
					actions[i] = torch::rand({1}).item<float>() * dqnOption.testOutput;
					LOG4CXX_INFO(logger, "random action for " << i << ": " << actions[i]);
					randomStep[i] ++;
					if (randomStep[i] > dqnOption.randomStep) {
						randomStep[i] = 0;
						noReward[i] = 0;
					}
				}
			}
		}
		auto stepResult = testEnv.step(actions, true);
		auto nextStateVec = std::get<0>(stepResult);
		auto rewardVec = std::get<1>(stepResult);
		auto doneVec = std::get<2>(stepResult);

		Stats::UpdateReward(statRewards, rewardVec);
		Stats::UpdateLen(statLens);

		for (int i = 0; i < batchSize; i ++) {
			if (doneVec[i]) {
				LOG4CXX_DEBUG(logger, "testEnv " << i << "done");
//				auto resetResult = env.reset(i);
				//udpate nextstatevec, target mask
//				std::copy(resetResult.begin(), resetResult.end(), nextStateVec.begin() + (offset * i));
				epCount ++;

				sumRewards[i] += statRewards[i];
				sumLens[i] += statLens[i];

//				testStater.update(statLens[i], statRewards[i]);
				tLogger.add_scalar("test/len", epCount, statLens[i]);
				tLogger.add_scalar("test/reward", epCount, statRewards[i]);
				LOG4CXX_INFO(logger, "test -----------> "<< i << " " << statLens[i] << ", " << statRewards[i]);
				statLens[i] = 0;
				statRewards[i] = 0;
//				stater.printCurStat();

				liveCounts[i] ++;
				if (liveCounts[i] >= dqnOption.donePerEp) {
					roundCount ++;
					LOG4CXX_INFO(logger, "Wrapper episode " << i << " ----------------------------> " << sumRewards[i]);
					tLogger.add_scalar("test/sumLen", roundCount, sumLens[i]);
					tLogger.add_scalar("test/sumReward", roundCount, sumRewards[i]);
//					sumStater.update(sumLens[i], sumRewards[i]);
					liveCounts[i] = 0;
					sumRewards[i] = 0;
					sumLens[i] = 0;
				}

			}

			if (dqnOption.randomHang) {
			//Good action should have reward
				if (rewardVec[i] < dqnOption.hangRewardTh) { //for float compare
					noReward[i] ++;
				}
			}
		}
		states = nextStateVec;
	}
}

template<typename NetType, typename OptimizerType, typename EnvType, typename PolicyType>
void A3CNStepTest<NetType, OptimizerType, EnvType, PolicyType>::train(const int epNum, bool batched) {
	LOG4CXX_INFO(logger, "batched " << batched);

	if (batched) {
		trainBatch(epNum);
	} else {
		trainStep(epNum);
	}
}

template<typename NetType, typename OptimizerType, typename EnvType, typename PolicyType>
void A3CNStepTest<NetType, OptimizerType, EnvType, PolicyType>::trainStep(const int epNum) {
	LOG4CXX_INFO(logger, "------------------------------> No batch");

	load();

	int step = 0;
	int epCount = 0;
	int updateNum = 0;

	std::vector<float> statRewards(batchSize, 0);
	std::vector<float> statLens(batchSize, 0);

	rollout.reset();
	std::vector<float> states = env.reset();
	while (updateNum < epNum) {
		step ++;
		updateNum ++;

		torch::Tensor stateTensor = torch::from_blob(states.data(), inputShape).div(dqnOption.inputScale).to(deviceType);
//		LOG4CXX_INFO(logger, "state input: " << stateTensor);
		std::vector<torch::Tensor> rc = bModel.forward(stateTensor);
		auto actionOutput = rc[0]; //TODO: detach?
		auto valueOutput = rc[1];
		auto actionProbs = torch::softmax(actionOutput, -1);
		std::vector<int64_t> actions = policy.getActions(actionProbs);

		auto stepResult = env.step(actions, false);
		auto nextStateVec = std::get<0>(stepResult);
		auto rewardVec = std::get<1>(stepResult);
		auto doneVec = std::get<2>(stepResult);

		Stats::UpdateReward(statRewards, rewardVec);
		Stats::UpdateLen(statLens);

		torch::Tensor doneTensor = torch::ones({batchSize, 1}).to(deviceType);
		for (int i = 0; i < batchSize; i ++) {
			if (doneVec[i]) {
				LOG4CXX_DEBUG(logger, "env " << i << "done");
				doneTensor[i] = 0;
//				auto resetResult = env.reset(i);
				//udpate nextstatevec, target mask
//				std::copy(resetResult.begin(), resetResult.end(), nextStateVec.begin() + (offset * i));
				epCount ++;

				LOG4CXX_INFO(logger, "update ep " << i << " = " << statLens[i] << ", " << statRewards[i]);
				tLogger.add_scalar("train/len", epCount, statLens[i]);
				tLogger.add_scalar("train/reward", epCount, statRewards[i]);
				LOG4CXX_INFO(logger, "" << updateNum << ": " << statLens[i] << ", " << statRewards[i]);
//				stater.update(statLens[i], statRewards[i]);
				statLens[i] = 0;
				statRewards[i] = 0;
//				stater.printCurStat();

			}
		}
		torch::Tensor rewardTensor = torch::from_blob(rewardVec.data(), {batchSize, 1}).to(deviceType).div(dqnOption.rewardScale).clamp(dqnOption.rewardMin, dqnOption.rewardMax);
		torch::Tensor actionTensor = torch::from_blob(actions.data(), {batchSize, 1}, longOpt).to(deviceType);

		rollout.put(valueOutput, actionOutput, actionTensor, rewardTensor, doneTensor);
		if (rollout.toUpdate()) {
			torch::Tensor nextValue;
			{
				torch::NoGradGuard guard;
				torch::Tensor nextInputTensor = torch::from_blob(nextStateVec.data(), inputShape).div(dqnOption.inputScale).to(deviceType);
				nextValue = bModel.forward(nextInputTensor)[1].detach();
			}
//			torch::Tensor loss = rollout.getLoss(nextValue, gamma, dqnOption.entropyCoef, dqnOption.valueCoef, stater, lossStater);
			auto losses = rollout.getLoss(nextValue, gamma, dqnOption.entropyCoef, dqnOption.valueCoef);
			auto loss = losses[0];
			auto actLoss = losses[1];
			auto valueLoss = losses[2];
			auto entropy = losses[3];

//			optimizer.zero_grad();
		    loss.backward();
			torch::nn::utils::clip_grad_norm_(bModel.parameters(), dqnOption.maxGradNormClip);
//		    optimizer.step();
			auto params = bModel.parameters();
			std::vector<torch::Tensor> grads;
			for (int i = 0; i < params.size(); i ++) {
				if (params[i].grad().numel() == 0) {
					grads.push_back(torch::zeros({0}));
				} else {
					grads.push_back(params[i].grad());
				}
			}
//			client->addGrad(grads);
			std::vector<torch::Tensor> tParams = tModel.parameters();
			for (int i = 0; i < params.size(); i ++) {
				if (grads[i].numel() == 0) {
					LOG4CXX_INFO(logger, "No grad of layer " << i);
					continue;
				}

				tParams[i].mutable_grad() = grads[i]; //TODO: grad valid?
			}

			optimizer.zero_grad();
			torch::nn::utils::clip_grad_norm_(tModel.parameters(), 0.1);
			optimizer.step();
			syncFromTarget();

		    rollout.reset();

		    if ((updateNum % dqnOption.logInterval) == 0) {
		    	tLogger.add_scalar("loss/loss", updateNum, loss.item<float>());
		    	tLogger.add_scalar("loss/actLoss", updateNum, actLoss.item<float>());
		    	tLogger.add_scalar("loss/valueLoss", updateNum, valueLoss.item<float>());
		    	tLogger.add_scalar("loss/entropy", updateNum, entropy.item<float>());
		    }

//		    if ((updateNum % dqnOption.gradSyncStep) == 0) {
//		    	client->sendGrad();
//		    }
//
//		    if ((updateNum % dqnOption.targetUpdateStep) == 0) {
//				client->syncTarget();
//			}

//		    updateNum ++;
		}

	    states = nextStateVec;
	}

	save();
}



template<typename NetType, typename OptimizerType, typename EnvType, typename PolicyType>
void A3CNStepTest<NetType, OptimizerType, EnvType, PolicyType>::trainBatch(const int epNum) {
	LOG4CXX_INFO(logger, "----------------------> batch");
	load();
	syncFromTarget();

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

//		torch::Tensor tmpValue;
//		bool isTmpValueSet = false;

		bModel.eval();
		for (step = 0; step < maxStep; step ++) {
			updateNum ++;

			torch::Tensor stateTensor = torch::from_blob(stateVec.data(), inputShape).div(dqnOption.inputScale).to(deviceType);
//			LOG4CXX_INFO(logger, "stateTensor in step: " << stateTensor.max() << stateTensor.mean());
			std::vector<torch::Tensor> rc = bModel.forward(stateTensor);
			auto actionProbs =  torch::softmax(rc[0], -1);
			std::vector<int64_t> actions = policy.getActions(actionProbs);
//			LOG4CXX_INFO(logger, "actions: " << actions);
//			LOG4CXX_INFO(logger, "state: " << stateTensor);


			auto stepResult = env.step(actions, false);
			auto nextStateVec = std::get<0>(stepResult);
			auto rewardVec = std::get<1>(stepResult);
			auto doneVec = std::get<2>(stepResult);

			Stats::UpdateReward(statRewards, rewardVec);
			Stats::UpdateLen(statLens);
			if (dqnOption.clipRewardStat) {
				Stats::UpdateReward(clipRewards, rewardVec, true, dqnOption.rewardMin, dqnOption.rewardMax);
			}

//			LOG4CXX_INFO(logger, "rewardVec" << step << ": " << rewardVec);
			std::vector<float> doneMaskVec(doneVec.size(), 1);
			for (int i = 0; i < doneVec.size(); i ++) {
//				LOG4CXX_INFO(logger, "dones: " << i << ": " << doneVec[i]);
				if (doneVec[i]) {
					doneMaskVec[i] = 0;
					epCount ++;

					sumRewards[i] += statRewards[i];
					sumLens[i] += statLens[i];

					tLogger.add_scalar("train/len", epCount, statLens[i]);
					tLogger.add_scalar("train/reward", epCount, statRewards[i]);
					LOG4CXX_INFO(logger, "" << epCount << ": " << statLens[i] << ", " << statRewards[i]);
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

		torch::Tensor lastStateTensor = torch::from_blob(stateVec.data(), inputShape).div(dqnOption.inputScale).to(deviceType);
//		LOG4CXX_INFO(logger, "stateTensor in last: " << lastStateTensor.max() << lastStateTensor.mean());

		auto rc = bModel.forward(lastStateTensor);
		auto lastValueTensor = rc[1].squeeze(-1);
//		LOG4CXX_INFO(logger, "lastValue: " << lastValueTensor);

		bModel.train();
//		optimizer.zero_grad();

		auto stateData = EnvUtils::FlattenVector(statesVec);
		torch::Tensor stateTensor = torch::from_blob(stateData.data(), batchInputShape).div(dqnOption.inputScale).to(deviceType);
//		if (dqnOption.isAtari) {
//			stateTensor = stateTensor.view({maxStep * batchSize, 4, 84, 84});
//		}
//		LOG4CXX_INFO(logger, "stateTensor in batch: " << stateTensor.max() << stateTensor.mean());
		auto output = bModel.forward(stateTensor);
		auto actionOutputTensor = output[0].squeeze(-1).view({maxStep, batchSize, -1}); //{maxstep * batch, actionNum, 1} -> {maxstep * batch, actionNum} -> {maxstep, batch, actionNum}
		auto valueTensor = output[1].squeeze(-1).view({maxStep, batchSize});
//		LOG4CXX_INFO(logger, "compare value " << tmpValue);
//		LOG4CXX_INFO(logger, "batch value: " << valueTensor);

		auto actionProbTensor = torch::softmax(actionOutputTensor, -1); //{maxstep, batch, actionNum}
		auto actionLogTensor = torch::log_softmax(actionOutputTensor, -1); //{maxStep, batch, actionNum}
		auto actionData = EnvUtils::FlattenVector(actionsVec);
		torch::Tensor actionTensor = torch::from_blob(actionData.data(), {maxStep, batchSize, 1}, longOpt).to(deviceType);
		auto maskData = EnvUtils::FlattenVector(donesVec);
		torch::Tensor maskTensor = torch::from_blob(maskData.data(), {maxStep, batchSize}).to(deviceType);
		auto rewardData = EnvUtils::FlattenVector(rewardsVec);
		torch::Tensor rewardTensor = torch::from_blob(rewardData.data(), {maxStep, batchSize}).to(deviceType);
		rewardTensor = rewardTensor.div(dqnOption.rewardScale).clamp(dqnOption.rewardMin, dqnOption.rewardMax);
//		if (dqnOption.normReward) {
//			rewardNorm.update(rewardTensor, maxStep * batchSize);
//			rewardTensor = (rewardTensor - rewardNorm.getMean()) / rewardNorm.getVar();
//		}
//		LOG4CXX_INFO(logger, "reward: " << rewardTensor);

		torch::Tensor returnTensor = torch::zeros({maxStep, batchSize}).to(deviceType);
		torch::Tensor qTensor = lastValueTensor;
//		LOG4CXX_INFO(logger, "qTensor begin: " << qTensor);
		for (int i = maxStep - 1; i >= 0; i --) {
			qTensor = qTensor * maskTensor[i] * gamma + rewardTensor[i];
//			LOG4CXX_INFO(logger, "qTensor" << i << ": " << qTensor);
//			LOG4CXX_INFO(logger, "maskTensor" << i << ": " << maskTensor[i]);
//			LOG4CXX_INFO(logger, "rewardTensor" << i << ": " << rewardTensor[i]);
//			LOG4CXX_INFO(logger, "row" << i << " qTensor: " << qTensor);
			returnTensor[i].copy_(qTensor);
		}
		if (dqnOption.normReward) {
			returnTensor = (returnTensor - returnTensor.mean()) / (returnTensor.std() + 1e-7);
		}
		returnTensor = returnTensor.detach();

		torch::Tensor entropyLoss = - (actionProbTensor * actionLogTensor).sum(-1).mean();

		auto advTensor = returnTensor - valueTensor;
		torch::Tensor valueLoss = torch::nn::functional::huber_loss(valueTensor, returnTensor);

//		LOG4CXX_INFO(logger, "actionLogTensor: " << actionLogTensor.sizes());
//		LOG4CXX_INFO(logger, "actionTensor: " << actionTensor.sizes());
		auto actPiTensor = actionLogTensor.gather(-1, actionTensor).squeeze(-1);
		torch::Tensor actLoss = - (actPiTensor * advTensor.detach()).mean();

		torch::Tensor loss = dqnOption.valueCoef * valueLoss + actLoss - dqnOption.entropyCoef * entropyLoss;

		auto lossV = loss.item<float>();
		auto aLossV = actLoss.item<float>();
		auto vLossV = valueLoss.item<float>();
		auto entropyV = entropyLoss.item<float>();

		optimizer.zero_grad();
		loss.backward();
		torch::nn::utils::clip_grad_norm_(bModel.parameters(), dqnOption.maxGradNormClip);
//		optimizer.step();

		auto params = bModel.parameters();

		/////////////////////////////////////////////////////////////

//		auto bParamDict = bModel.named_parameters();
//		for (auto& bParamItem: bParamDict) {
//			auto& bParam = bParamItem.value();
//			LOG4CXX_INFO(logger, "bParam \n" << bParam);
//			LOG4CXX_INFO(logger, "bParam  grad \n" << bParam.grad());
//		}
		auto bParams = bModel.parameters();
		auto tParams = tModel.parameters();
		for (int pIndex = 0; pIndex < bParams.size(); pIndex ++) {
			tParams[pIndex].mutable_grad() = bParams[pIndex].grad();
		}
//		for (auto& param: tModel.parameters()) {
//			LOG4CXX_INFO(logger, "tModel grad: \n" << param.grad());
//		}

//		torch::nn::utils::clip_grad_norm_(tModel.parameters(), 0.1);
		optimizer.step();

		syncFromTarget();

//		client->addGrad(grads);

		LOG4CXX_INFO(logger, "loss" << updateNum << ": " << lossV
			<< ", " << vLossV << ", " << aLossV << ", " << entropyV);
		if ((updateNum % dqnOption.logInterval) == 0) {
			tLogger.add_scalar("loss/loss", updateNum, lossV);
			tLogger.add_scalar("loss/vLoss", updateNum, vLossV);
			tLogger.add_scalar("loss/aLoss", updateNum, aLossV);
			tLogger.add_scalar("loss/entropy", updateNum, entropyV);
		}
//
//		if ((updateNum % dqnOption.gradSyncStep) == 0) {
//			client->sendGrad();
//		}
//		if ((updateNum % dqnOption.targetUpdateStep) == 0) {
//			client->syncTarget();
//		}

	}

	save();
}

template<typename NetType, typename OptimizerType, typename EnvType, typename PolicyType>
void A3CNStepTest<NetType, OptimizerType, EnvType, PolicyType>::save() {
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

template<typename NetType, typename OptimizerType, typename EnvType, typename PolicyType>
void A3CNStepTest<NetType, OptimizerType, EnvType, PolicyType>::saveByReward(float reward) {
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

template<typename NetType, typename OptimizerType, typename EnvType, typename PolicyType>
void A3CNStepTest<NetType, OptimizerType, EnvType, PolicyType>::load() {
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


template<typename NetType, typename OptimizerType, typename EnvType, typename PolicyType>
void A3CNStepTest<NetType, OptimizerType, EnvType, PolicyType>::syncFromTarget() {
	torch::NoGradGuard guard;

	auto paramDict = bModel.named_parameters();
	auto buffDict = bModel.named_buffers();
	auto targetParamDict = tModel.named_parameters();
	auto targetBuffDict = tModel.named_buffers();

	for (const auto& item: targetParamDict) {
		const auto& key = item.key();
		const auto param = item.value();
		auto& origParam = paramDict[key];

		origParam.mul_(1 - dqnOption.tau);
		origParam.add_(param, dqnOption.tau);
	}

	for (const auto& item: targetBuffDict) {
		const auto& key = item.key();
		const auto& buff = item.value();
		auto& origBuff = buffDict[key];

		origBuff.mul_(1 - dqnOption.tau);
		origBuff.add_(buff, dqnOption.tau);
	}
	LOG4CXX_INFO(logger, "target network synched");
}



#endif /* INC_ALG_A3CTEST_HPP_ */
