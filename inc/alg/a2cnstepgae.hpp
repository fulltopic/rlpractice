/*
 * a2cnstepgae.hpp
 *
 *  Created on: Jul 6, 2021
 *      Author: zf
 */

#ifndef INC_ALG_A2CNSTEPGAE_HPP_
#define INC_ALG_A2CNSTEPGAE_HPP_


#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <vector>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/a2cstatestore.h"
#include "gymtest/utils/stats.h"
#include "gymtest/utils/lossstats.h"
#include "gymtest/utils/a2cnstore.h"
#include "dqnoption.h"

#include "gymtest/utils/inputnorm.h"

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
class A2CNStepGae {
private:
	NetType& bModel;
//	NetType& tModel;
	EnvType& env;
	EnvType& testEnv;
	PolicyType& policy;
	OptimizerType& optimizer;
	const torch::Device deviceType;
	const at::IntArrayRef inputShape;

	const DqnOption dqnOption;
	Stats stater;
	Stats testStater;
	Stats sumStater;
	LossStats lossStater;

	const int batchSize;
	int offset = 0;

	uint32_t updateNum = 0;
	const int updateTargetGap; //TODO

	A2CNStorage rollout;
	const uint32_t maxStep;
	uint32_t qIndex = 0;

	const float entropyCoef;

	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("a2c1stepq");
	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	torch::nn::HuberLoss huberLossComputer = torch::nn::HuberLoss();
	torch::nn::MSELoss mseLossComputer = torch::nn::MSELoss();

	InputNorm rewardNorm;



	void save();

//	void trainStep(const int epNum); //no batched
//	void trainBatch(const int epNum); //batched

public:
	const float gamma;
	const int testEp = 16;

	A2CNStepGae(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, int stepSize, DqnOption option);
	~A2CNStepGae() = default;
	A2CNStepGae(const A2CNStepGae& ) = delete;

	void train(const int epNum);

	void test(const int batchSize, const int epochNum);
	void load();
};

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
A2CNStepGae<NetType, EnvType, PolicyType, OptimizerType>::A2CNStepGae(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer,
		int stepSize, const DqnOption iOption):
	bModel(behaviorModel),
//	tModel(trainModel),
	env(iEnv),
	testEnv(tEnv),
	policy(iPolicy),
	optimizer(iOptimizer),
	dqnOption(iOption),
	deviceType(iOption.deviceType),
	inputShape(iOption.inputShape),
	gamma(iOption.gamma),
	stater(iOption.statPathPrefix + "_stat.txt", iOption.statCap),
	testStater(iOption.statPathPrefix + "_test.txt", iOption.statCap),
	sumStater(iOption.statPathPrefix + "_sum.txt", iOption.statCap),
	lossStater(iOption.statPathPrefix + "_loss.txt"),
	batchSize(iOption.batchSize),
	updateTargetGap(iOption.targetUpdate),
	rollout(iOption.deviceType, stepSize),
	maxStep(stepSize),
	entropyCoef(iOption.entropyCoef),
	rewardNorm(iOption.deviceType)
{
	offset = 1;
	for (int i = 1; i < inputShape.size(); i ++) {
		offset *= inputShape[i];
	}
//
//	std::string statPath = iOption.statPathPrefix + "_stat.txt";
//	std::string testStatPath = iOption.statPathPrefix + "_test.txt";
//	std::string lossStatPath = iOption.statPathPrefix + "_loss.txt";
//	stater = Stats(statPath);
//	testStater = Stats(testStatPath);
//	lossStater = LossStats(lossStatPath);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CNStepGae<NetType, EnvType, PolicyType, OptimizerType>::test(const int batchSize, const int epochNum) {
	LOG4CXX_INFO(logger, "To test " << epochNum << "episodes");
	if (!dqnOption.toTest) {
		return;
	}

	int epCount = 0;
	std::vector<float> statRewards(batchSize, 0);
	std::vector<float> statLens(batchSize, 0);

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

				testStater.update(statLens[i], statRewards[i]);
				statLens[i] = 0;
				statRewards[i] = 0;
//				stater.printCurStat();
				LOG4CXX_INFO(logger, "test -----------> " << testStater);

			}
		}
		states = nextStateVec;
	}
}



template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CNStepGae<NetType, EnvType, PolicyType, OptimizerType>::train(const int epNum) {
	LOG4CXX_INFO(logger, "----------------------> batch");
	load();

	int step = 0;
	int epCount = 0;
	int updateNum = 0;

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


	std::vector<float> stateVec = env.reset();
	while (updateNum < epNum) {
		std::vector<std::vector<float>> statesVec;
		std::vector<std::vector<float>> rewardsVec;
		std::vector<std::vector<float>> donesVec;
		std::vector<std::vector<long>> actionsVec;

//		std::vector<torch::Tensor> nextValues;

//		torch::Tensor tmpValue;
//		bool isTmpValueSet = false;

		bModel.eval();
		for (step = 0; step < maxStep; step ++) {
			torch::Tensor stateTensor = torch::from_blob(stateVec.data(), inputShape).div(dqnOption.inputScale).to(deviceType);
//			LOG4CXX_INFO(logger, "stateTensor in step: " << stateTensor.max() << stateTensor.mean());
			std::vector<torch::Tensor> rc = bModel.forward(stateTensor);
			auto actionProbs =  torch::softmax(rc[0], -1);
			std::vector<int64_t> actions = policy.getActions(actionProbs);
//			nextValues.push_back(rc[1].squeeze(-1));
//			if (!isTmpValueSet) {
//				tmpValue = rc[1];
//				isTmpValueSet = true;
//			}

			auto stepResult = env.step(actions, false);
			auto nextStateVec = std::get<0>(stepResult);
			auto rewardVec = std::get<1>(stepResult);
			auto doneVec = std::get<2>(stepResult);

			Stats::UpdateReward(statRewards, rewardVec);
			Stats::UpdateLen(statLens);

//			LOG4CXX_INFO(logger, "rewardVec" << step << ": " << rewardVec);
			std::vector<float> doneMaskVec(doneVec.size(), 1);
			for (int i = 0; i < doneVec.size(); i ++) {
//				LOG4CXX_INFO(logger, "dones: " << i << ": " << doneVec[i]);
				if (doneVec[i]) {
					doneMaskVec[i] = 0;
					epCount ++;

					sumRewards[i] += statRewards[i];
					sumLens[i] += statLens[i];

					stater.update(statLens[i], statRewards[i]);
					statLens[i] = 0;
					statRewards[i] = 0;
					LOG4CXX_INFO(logger, stater);

					if (dqnOption.multiLifes) {
						liveCounts[i] ++;
						if (liveCounts[i] >= dqnOption.donePerEp) {
							LOG4CXX_INFO(logger, "Wrapper episode " << i << " ----------------------------> " << sumRewards[i]);
							sumStater.update(sumLens[i], sumRewards[i]);

							liveCounts[i] = 0;
							sumRewards[i] = 0;
							sumLens[i] = 0;
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
		optimizer.zero_grad();

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
		torch::Tensor gaeTensor =  torch::zeros({maxStep, batchSize}).to(deviceType);
		torch::Tensor qTensor = lastValueTensor;
		torch::Tensor gaeReturn = torch::zeros({batchSize}).to(deviceType);
		torch::Tensor plainReturn = torch::zeros({batchSize}).to(deviceType); //TODO: deviceType as creator parameter
		plainReturn.copy_(lastValueTensor);
//		LOG4CXX_INFO(logger, "qTensor begin: " << qTensor.sizes());
//		LOG4CXX_INFO(logger, "nextValues: " << nextValues[0].sizes());
//		LOG4CXX_INFO(logger, "mask: " << maskTensor[0].sizes());
//		LOG4CXX_INFO(logger, "returnTensor: " << returnTensor[0].sizes());

		for (int i = maxStep - 1; i >= 0; i --) {
			torch::Tensor delta = rewardTensor[i] + dqnOption.gamma * qTensor * maskTensor[i] - valueTensor[i];
			gaeReturn = delta + dqnOption.ppoLambda * dqnOption.gamma * gaeReturn * maskTensor[i];

			plainReturn = rewardTensor[i] + dqnOption.gamma * plainReturn * maskTensor[i];

			returnTensor[i].copy_(plainReturn);
			gaeTensor[i].copy_(gaeReturn);
			qTensor = valueTensor[i];

//			qTensor = qTensor * maskTensor[i] * gamma + rewardTensor[i];
//			LOG4CXX_INFO(logger, "qTensor" << i << ": " << qTensor);
//			LOG4CXX_INFO(logger, "maskTensor" << i << ": " << maskTensor[i]);
//			LOG4CXX_INFO(logger, "rewardTensor" << i << ": " << rewardTensor[i]);
//			LOG4CXX_INFO(logger, "row" << i << " qTensor: " << qTensor);
//			returnTensor[i].copy_(qTensor);
		}
		if (dqnOption.normReward) {
			gaeTensor = (gaeTensor - gaeTensor.mean()) / (gaeTensor.std() + 1e-7);
			returnTensor = (returnTensor - returnTensor.mean()) / (returnTensor.std() + 1e-7);
		}
		returnTensor = returnTensor.detach();
		gaeTensor = gaeTensor.detach();

		//Computing loss
		torch::Tensor entropyLoss = - (actionProbTensor * actionLogTensor).sum(-1).mean();

//		auto advTensor = returnTensor - valueTensor;
//		LOG4CXX_INFO(logger, "returnTensor: " << returnTensor);
//		LOG4CXX_INFO(logger, "valueTensor: " << valueTensor);
//		torch::Tensor valueLoss = advTensor.pow(2).mean();
//		torch::nn::SmoothL1Loss lossComputer = torch::nn::SmoothL1Loss();
		//TODO: What huberLossComputer(x, y) did? Should it be huberLossComputer->forward(x, y)?
//		torch::Tensor valueLoss = huberLossComputer(valueTensor, returnTensor);
		torch::Tensor valueLoss = torch::nn::functional::mse_loss(valueTensor, returnTensor);

//		LOG4CXX_INFO(logger, "actionLogTensor: " << actionLogTensor.sizes());
//		LOG4CXX_INFO(logger, "actionTensor: " << actionTensor.sizes());
		auto actPiTensor = actionLogTensor.gather(-1, actionTensor).squeeze(-1);
//		LOG4CXX_INFO(logger, "actionLogTensor: " << actionLogTensor);
//		LOG4CXX_INFO(logger, "actPiTensor: " << actPiTensor);
//		LOG4CXX_INFO(logger, "actionTensor: " << actionTensor);
//		LOG4CXX_INFO(logger, "advTensor: " << advTensor);
		torch::Tensor actLoss = - (actPiTensor * gaeTensor).mean();

		torch::Tensor loss = dqnOption.valueCoef * valueLoss + actLoss - dqnOption.entropyCoef * entropyLoss;

		auto lossV = loss.item<float>();
		auto aLossV = actLoss.item<float>();
		auto vLossV = valueLoss.item<float>();
		auto entropyV = entropyLoss.item<float>();

		loss.backward();
//		torch::nn::utils::clip_grad_value_(bModel.parameters(), 1);
		torch::nn::utils::clip_grad_norm_(bModel.parameters(), dqnOption.maxGradNormClip);
		optimizer.step();

		LOG4CXX_INFO(logger, "loss" << updateNum << ": " << lossV
			<< ", " << vLossV << ", " << aLossV << ", " << entropyV);
		auto curState = stater.getCurState();
		lossStater.update({lossV, vLossV, aLossV, entropyV,
			vLossV * dqnOption.valueCoef, entropyV * dqnOption.entropyCoef * (-1),
			curState[0], curState[1]});

		updateNum ++;
	}

	save();
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CNStepGae<NetType, EnvType, PolicyType, OptimizerType>::save() {
	if (!dqnOption.saveModel) {
		return;
	}

	std::string modelPath = dqnOption.savePathPrefix + "_model.pt";
	torch::serialize::OutputArchive outputArchive;
	bModel.save(outputArchive);
	outputArchive.save_to(modelPath);
	LOG4CXX_INFO(logger, "Save model into " << modelPath);

	std::string optPath = dqnOption.savePathPrefix + "_optimizer.pt";
	torch::serialize::OutputArchive optimizerArchive;
	optimizer.save(optimizerArchive);
	optimizerArchive.save_to(optPath);
	LOG4CXX_INFO(logger, "Save optimizer into " << optPath);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CNStepGae<NetType, EnvType, PolicyType, OptimizerType>::load() {
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
		optimizer.load(opInChive);
		LOG4CXX_INFO(logger, "Load optimizer from " << optPath);
	}

}


#endif /* INC_ALG_A2CNSTEPGAE_HPP_ */
