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

#include <tensorboard_logger.h>

#include <vector>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/a2cstatestore.h"
#include "gymtest/utils/stats.h"
#include "gymtest/utils/lossstats.h"
#include "gymtest/utils/a2cnstore.h"
#include "gymtest/utils/inputnorm.h"
#include "alg/utils/dqnoption.h"
#include "alg/utils/algtester.hpp"
#include "alg/utils/utils.hpp"

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
	TensorBoardLogger tLogger;

	const int batchSize;
	const uint32_t maxStep;
	uint32_t updateNum = 0;


	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("a2c1stepq");
	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);

	AlgTester<NetType, EnvType, PolicyType> tester;


	void save();


public:
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
	env(iEnv),
	testEnv(tEnv),
	policy(iPolicy),
	optimizer(iOptimizer),
	dqnOption(iOption),
	deviceType(iOption.deviceType),
	inputShape(iOption.inputShape),
	batchSize(iOption.batchSize),
	maxStep(stepSize),
	tLogger(iOption.tensorboardLogPath.c_str()),
	tester(behaviorModel, tEnv, iPolicy, iOption, tLogger)
{

}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CNStepGae<NetType, EnvType, PolicyType, OptimizerType>::test(const int batchSize, const int epochNum) {
	tester.testAC();
}



template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CNStepGae<NetType, EnvType, PolicyType, OptimizerType>::train(const int epNum) {
	LOG4CXX_INFO(logger, "----------------------> batch");
	load();

	int step = 0;
	int epCount = 0;
	int roundCount = 0;

	std::vector<int64_t> batchInputShape;
	if (dqnOption.isAtari) {
		batchInputShape.push_back(maxStep * batchSize);
		for (int i = 1; i < inputShape.size(); i ++) {
			batchInputShape.push_back(inputShape[i]);
		}
	} else {
		batchInputShape.push_back(maxStep);
		for (int i = 0; i < inputShape.size(); i ++) {
			batchInputShape.push_back(inputShape[i]);
		}
	}


	std::vector<std::vector<float>> statesVec;
	std::vector<std::vector<float>> rewardsVec;
	std::vector<std::vector<float>> donesVec;
	std::vector<std::vector<long>> actionsVec;

	std::vector<float> statRewards(batchSize, 0);
	std::vector<float> statLens(batchSize, 0);
	std::vector<int> liveCounts(batchSize, 0);
	std::vector<float> sumRewards(batchSize, 0);
	std::vector<float> sumLens(batchSize, 0);


	std::vector<float> stateVec = env.reset();
	while (updateNum < epNum) {

		statesVec.clear();
		rewardsVec.clear();
		donesVec.clear();
		actionsVec.clear();

		{
			torch::NoGradGuard guard;

		for (step = 0; step < maxStep; step ++) {
			updateNum ++;

			torch::Tensor stateTensor = torch::from_blob(stateVec.data(), inputShape).div(dqnOption.inputScale).to(deviceType);
//			LOG4CXX_INFO(logger, "stateTensor in step: " << stateTensor.max() << stateTensor.mean());
			std::vector<torch::Tensor> rc = bModel.forward(stateTensor);
			auto actionProbs =  torch::softmax(rc[0], -1);
			std::vector<int64_t> actions = policy.getActions(actionProbs);


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

					tLogger.add_scalar("train/len", epCount, statLens[i]);
					tLogger.add_scalar("train/reward", epCount, statRewards[i]);
					LOG4CXX_INFO(logger, "ep " << updateNum << ": " << statLens[i] << ", " << statRewards[i]);
					auto curReward = statRewards[i];

					statLens[i] = 0;
					statRewards[i] = 0;

					if (dqnOption.multiLifes) {
						liveCounts[i] ++;
						if (liveCounts[i] >= dqnOption.donePerEp) {
							roundCount ++;
							LOG4CXX_INFO(logger, "Wrapper episode " << i << " ----------------------------> " << sumRewards[i]);
							tLogger.add_scalar("train/sumLen", roundCount, sumLens[i]);
							tLogger.add_scalar("train/sumReward", roundCount, sumRewards[i]);
							auto curSumReward = sumRewards[i];

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
		} //nogradguard

		//Get last V
		torch::Tensor lastStateTensor = torch::from_blob(stateVec.data(), inputShape).div(dqnOption.inputScale).to(deviceType);
//		LOG4CXX_INFO(logger, "stateTensor in last: " << lastStateTensor.max() << lastStateTensor.mean());
		auto rc = bModel.forward(lastStateTensor);
		auto lastValueTensor = rc[1].squeeze(-1).detach();
//		LOG4CXX_INFO(logger, "lastValue: " << lastValueTensor);

		auto stateData = EnvUtils::FlattenVector(statesVec);
		torch::Tensor stateTensor = torch::from_blob(stateData.data(), batchInputShape).div(dqnOption.inputScale).to(deviceType);
//		LOG4CXX_INFO(logger, "stateTensor in batch: " << stateTensor.max() << stateTensor.mean());
		auto output = bModel.forward(stateTensor);
		auto actionOutputTensor = output[0].squeeze(-1).view({maxStep, batchSize, -1}); //{maxstep * batch, actionNum, 1} -> {maxstep * batch, actionNum} -> {maxstep, batch, actionNum}
		torch::Tensor valueTensor = output[1].squeeze(-1).view({maxStep, batchSize});
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
		if (dqnOption.tdValue) {
			returnTensor = gaeTensor + valueTensor; //from baseline3.
		}
		returnTensor = returnTensor.detach();
		gaeTensor = gaeTensor.detach();

		//Computing loss
		torch::Tensor entropyLoss = - (actionProbTensor * actionLogTensor).sum(-1).mean();

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



		optimizer.zero_grad();
		loss.backward();
		torch::nn::utils::clip_grad_norm_(bModel.parameters(), dqnOption.maxGradNormClip);
		optimizer.step();


		if ((updateNum % dqnOption.logInterval) == 0) {
			auto lossV = loss.item<float>();
			auto aLossV = actLoss.item<float>();
			auto vLossV = valueLoss.item<float>();
			auto entropyV = entropyLoss.item<float>();
			auto valueV = valueTensor.mean().item<float>();


			tLogger.add_scalar("loss/loss", updateNum, lossV);
			tLogger.add_scalar("loss/vLoss", updateNum, vLossV);
			tLogger.add_scalar("loss/aLoss", updateNum, aLossV);
			tLogger.add_scalar("loss/entropy", updateNum, entropyV);
			tLogger.add_scalar("loss/v", updateNum, valueV);
		}

		if ((updateNum % dqnOption.testGapEp) == 0) {
			if (dqnOption.toTest) {
				test(dqnOption.testBatch, dqnOption.testEp);
			}
		}
	}

	save();
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CNStepGae<NetType, EnvType, PolicyType, OptimizerType>::save() {
	if (!dqnOption.saveModel) {
		return;
	}

	AlgUtils::SaveModel(bModel, optimizer, dqnOption.savePathPrefix, logger);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CNStepGae<NetType, EnvType, PolicyType, OptimizerType>::load() {
	if (!dqnOption.loadModel) {
		return;
	}

	AlgUtils::LoadModel(bModel, optimizer, dqnOption.loadOptimizer, dqnOption.loadPathPrefix, logger);
}


#endif /* INC_ALG_A2CNSTEPGAE_HPP_ */
