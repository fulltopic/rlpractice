/*
 * a2cnstepnorm.hpp
 *
 *  Created on: May 14, 2021
 *      Author: zf
 */

#ifndef INC_ALG_A2CNSTEPNORM_HPP_
#define INC_ALG_A2CNSTEPNORM_HPP_



#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <vector>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/a2cstatestore.h"
#include "gymtest/utils/stats.h"
#include "gymtest/utils/lossstats.h"
#include "gymtest/utils/a2cnstore.h"
#include "gymtest/utils/inputnorm.h"
#include "utils/dqnoption.h"

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
class A2CNStepNorm {
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
	LossStats lossStater;

	const int batchSize;
	int offset = 0;

	uint32_t updateNum = 0;
	const int updateTargetGap; //TODO

	A2CNStorage rollout;
	const uint32_t maxStep;
	uint32_t qIndex = 0;

	const float entropyCoef;

	InputNorm rewardNorm;

	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("a2cbatchnorm");
	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);

	void test(const int batchSize, const int epochNum);

	void save();
	void load();

//	void trainStep(const int epNum); //no batched
//	void trainBatch(const int epNum); //batched

public:
	const float gamma;
	const int testEp = 16;

	A2CNStepNorm(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, int stepSize, DqnOption option);
	~A2CNStepNorm() = default;
	A2CNStepNorm(const A2CNStepNorm& ) = delete;

	void train(const int epNum);
};

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
A2CNStepNorm<NetType, EnvType, PolicyType, OptimizerType>::A2CNStepNorm(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer,
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
	stater(iOption.statPathPrefix + "_stat.txt"),
	testStater(iOption.statPathPrefix + "_test.txt"),
	lossStater(iOption.statPathPrefix + "_loss.txt"),
	batchSize(iOption.batchSize),
	updateTargetGap(iOption.targetUpdate),
	rollout(iOption.deviceType, stepSize),
	maxStep(stepSize),
	entropyCoef(iOption.entropyCoef),
	rewardNorm(deviceType)
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
void A2CNStepNorm<NetType, EnvType, PolicyType, OptimizerType>::test(const int batchSize, const int epochNum) {
	if (!dqnOption.toTest) {
		return;
	}

	int epCount = 0;
	std::vector<float> statRewards(batchSize, 0);
	std::vector<float> statLens(batchSize, 0);

	torch::NoGradGuard guard;
	std::vector<float> states = testEnv.reset();
	while (epCount < epochNum) {
		torch::Tensor stateTensor = torch::from_blob(states.data(), inputShape).to(deviceType);
		if (dqnOption.isAtari) {
			stateTensor = stateTensor.div(255);
		}
		std::vector<torch::Tensor> rc = bModel.forward(stateTensor);
		auto actionOutput = rc[0]; //TODO: detach?
		auto valueOutput = rc[1];
		auto actionProbs = torch::softmax(actionOutput, -1);
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

//Sparse reward made the running reward too stable: small mean and small var, leading to big processed rewards
template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CNStepNorm<NetType, EnvType, PolicyType, OptimizerType>::train(const int epNum) {
	LOG4CXX_INFO(logger, "----------------------> batch");
	load();

	int step = 0;
	int epCount = 0;
	int updateNum = 0;

	std::vector<int64_t> batchInputShape(1 + inputShape.size(), 0);
	batchInputShape[0] = maxStep;
	for (int i = 1; i < batchInputShape.size(); i ++) {
		batchInputShape[i] = inputShape[i - 1];
	}

	std::vector<float> statRewards(batchSize, 0);
	std::vector<float> statLens(batchSize, 0);

	std::vector<float> stateVec = env.reset();
	while (epCount < epNum) {
		std::vector<std::vector<float>> statesVec;
		std::vector<std::vector<float>> rewardsVec;
		std::vector<std::vector<float>> donesVec;
		std::vector<std::vector<long>> actionsVec;

		bModel.eval();
		for (step = 0; step < maxStep; step ++) {
			torch::Tensor stateTensor = torch::from_blob(stateVec.data(), inputShape).div(dqnOption.inputScale).to(deviceType);

//			inputNorm.update(stateTensor, 1);
//			stateTensor = (stateTensor - inputNorm.getMean()) / inputNorm.getVar();
//			LOG4CXX_INFO(logger, "state Tensor " << "\n" << stateTensor);

			std::vector<torch::Tensor> rc = bModel.forward(stateTensor);
			auto actionProbs =  torch::softmax(rc[0], -1);
			std::vector<int64_t> actions = policy.getActions(actionProbs);

			auto stepResult = env.step(actions, false);
			auto nextStateVec = std::get<0>(stepResult);
			auto rewardVec = std::get<1>(stepResult);
			auto doneVec = std::get<2>(stepResult);

			Stats::UpdateReward(statRewards, rewardVec);
			Stats::UpdateLen(statLens);

			std::vector<float> doneMaskVec(doneVec.size(), 1);
			for (int i = 0; i < doneVec.size(); i ++) {
//				LOG4CXX_INFO(logger, "dones: " << i << ": " << doneVec[i]);
				if (doneVec[i]) {
					doneMaskVec[i] = 0;
					epCount ++;

					stater.update(statLens[i], statRewards[i]);
					statLens[i] = 0;
					statRewards[i] = 0;
					LOG4CXX_INFO(logger, stater);
				}
			}

			statesVec.push_back(stateVec);
			rewardsVec.push_back(rewardVec);
			donesVec.push_back(doneMaskVec);
			actionsVec.push_back(actions);

			stateVec = nextStateVec;
		}

		torch::Tensor lastStateTensor = torch::from_blob(stateVec.data(), inputShape).div(dqnOption.inputScale).to(deviceType);
//		lastStateTensor = (lastStateTensor - inputNorm.getMean()) / inputNorm.getVar();
		auto rc = bModel.forward(lastStateTensor);
		auto lastValueTensor = rc[1].squeeze(-1);

		bModel.train();
		torch::Tensor stateTensor = torch::from_blob(EnvUtils::FlattenVector(statesVec).data(), batchInputShape).div(dqnOption.inputScale).to(deviceType);
//		stateTensor = (stateTensor - inputNorm.getMean()) / inputNorm.getVar();

		if (dqnOption.isAtari) {
			stateTensor = stateTensor.view({maxStep * batchSize, 4, 84, 84});
		}
		auto output = bModel.forward(stateTensor);
		auto actionOutputTensor = output[0].squeeze(-1).view({maxStep, batchSize, -1}); //{maxstep * batch, actionNum, 1} -> {maxstep * batch, actionNum} -> {maxstep, batch, actionNum}
		auto valueTensor = output[1].squeeze(-1).view({maxStep, batchSize});
		auto actionProbTensor = torch::softmax(actionOutputTensor, -1); //{maxstep, batch, actionNum}
		auto actionLogTensor = torch::log_softmax(actionOutputTensor, -1); //{maxStep, batch, actionNum}
		torch::Tensor actionTensor = torch::from_blob(EnvUtils::FlattenVector(actionsVec).data(), {maxStep, batchSize, 1}, longOpt).to(deviceType);
		torch::Tensor maskTensor = torch::from_blob(EnvUtils::FlattenVector(donesVec).data(), {maxStep, batchSize}).to(deviceType);
		torch::Tensor rewardTensor = torch::from_blob(EnvUtils::FlattenVector(rewardsVec).data(), {maxStep, batchSize}).to(deviceType);
		rewardNorm.update(rewardTensor, batchSize);
		LOG4CXX_INFO(logger, "reward before: " << rewardTensor);
		rewardTensor = (rewardTensor - rewardNorm.getMean()) / rewardNorm.getVar();
		LOG4CXX_INFO(logger, "mean = " << rewardNorm.getMean() << ", var = " << rewardNorm.getVar());
		LOG4CXX_INFO(logger, "reward: " << rewardTensor);
//		rewardTensor = rewardTensor.div(dqnOption.rewardScale).clamp(dqnOption.rewardMin, dqnOption.rewardMax);

		torch::Tensor returnTensor = torch::zeros({maxStep, batchSize}).to(deviceType);
		torch::Tensor qTensor = lastValueTensor;
		for (int i = maxStep - 1; i >= 0; i --) {
			qTensor = qTensor * maskTensor[i] * gamma + rewardTensor[i];
//			LOG4CXX_INFO(logger, "row" << i << " qTensor: " << qTensor);
			returnTensor[i].copy_(qTensor);
		}
//		LOG4CXX_INFO(logger, "returnTensor: " << returnTensor);


		torch::Tensor entropyLoss = - (actionProbTensor * actionLogTensor).sum(-1).mean();

		auto advTensor = returnTensor - valueTensor;
		torch::Tensor valueLoss = advTensor.pow(2).mean() * 0.5;


//		LOG4CXX_INFO(logger, "actionOutput: " << actionOutputTensor);
//		LOG4CXX_INFO(logger, "actionProb: " << actionProbTensor);
//		LOG4CXX_INFO(logger, "actionLogTensor: " << actionLogTensor);


//		LOG4CXX_INFO(logger, "actionLogTensor: " << actionLogTensor.sizes());
//		LOG4CXX_INFO(logger, "actionTensor: " << actionTensor.sizes());
		auto actPiTensor = actionLogTensor.gather(-1, actionTensor).squeeze(-1);
//		LOG4CXX_INFO(logger, "actPiTensor: " << actPiTensor);
//		LOG4CXX_INFO(logger, "actionTensor: " << actionTensor);
//		LOG4CXX_INFO(logger, "advTensor: " << advTensor);
		torch::Tensor actLoss = - (actPiTensor * advTensor.detach()).mean();

		torch::Tensor loss = valueLoss + actLoss - dqnOption.entropyCoef * entropyLoss;
		LOG4CXX_INFO(logger, "loss" << epCount << ": " << loss.item<float>()
			<< ", " << valueLoss.item<float>() << ", " << actLoss.item<float>() << ", " << entropyLoss.item<float>());
		lossStater.update({ loss.item<float>(), valueLoss.item<float>(), actLoss.item<float>(), entropyLoss.item<float>()});

		optimizer.zero_grad();
		loss.backward();
		torch::nn::utils::clip_grad_value_(bModel.parameters(), 1);
		optimizer.step();
	}

	save();
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CNStepNorm<NetType, EnvType, PolicyType, OptimizerType>::save() {
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
void A2CNStepNorm<NetType, EnvType, PolicyType, OptimizerType>::load() {
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


#endif /* INC_ALG_A2CNSTEPNORM_HPP_ */
