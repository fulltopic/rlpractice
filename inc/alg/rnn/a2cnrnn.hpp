/*
 * a2cnrnn.hpp
 *
 *  Created on: Jan 27, 2022
 *      Author: zf
 */

#ifndef INC_ALG_A2CNRNN_HPP_
#define INC_ALG_A2CNRNN_HPP_



#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <tensorboard_logger.h>

#include <vector>
#include <queue>
#include <algorithm>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/a2cstatestore.h"
#include "gymtest/utils/stats.h"
#include "gymtest/utils/lossstats.h"
#include "gymtest/utils/a2cnstore.h"
#include "gymtest/utils/inputnorm.h"
#include "alg/utils/dqnoption.h"
#include "alg/utils/algtester.hpp"
#include "alg/utils/utils.hpp"


struct Episode {
	std::vector<float> states; //TODO: Not to vector of vector, but insert at the end
	std::vector<int64_t> actions;
	std::vector<float> rewards;
	std::vector<float> returns;
};

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
class A2CNRNN {
private:
	NetType& bModel;
	EnvType& env;
	EnvType& testEnv;
	PolicyType& policy;
	OptimizerType& optimizer;
	const torch::Device deviceType;
	const at::IntArrayRef inputShape;

	const DqnOption dqnOption;

	TensorBoardLogger tLogger;

	const int batchSize;
	int64_t dataSize = 1;

	uint32_t updateNum = 0;
	uint32_t testEpCount = 0;
	const int updateTargetGap; //TODO

	const uint32_t maxStep;

	const float entropyCoef;

	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("a2c1stepq");
	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);

	InputNorm rewardNorm;

	float maxAveReward;
	float maxSumReward;

	AlgTester<NetType, EnvType, PolicyType> tester;
	std::vector<torch::Tensor> testHiddenStates;
	std::queue<Episode> eps;


	void save();
	void saveByReward(float reward);

	void trainBatch(const int epNum); //batched

public:
	const float gamma;
	const int testEp = 16;

	A2CNRNN(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, int stepSize, DqnOption option);
	~A2CNRNN() = default;
	A2CNRNN(const A2CNRNN& ) = delete;

	void train(const int epNum); //batched
	void test(const int batchSize, const int epochNum);
	void load();
};

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
A2CNRNN<NetType, EnvType, PolicyType, OptimizerType>::A2CNRNN(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer,
		int stepSize, const DqnOption iOption):
	bModel(behaviorModel),
	env(iEnv),
	testEnv(tEnv),
	policy(iPolicy),
	optimizer(iOptimizer),
	dqnOption(iOption),
	deviceType(iOption.deviceType),
	inputShape(iOption.inputShape),
	gamma(iOption.gamma),
	tLogger(iOption.tensorboardLogPath.c_str()),
	batchSize(iOption.batchSize),
	updateTargetGap(iOption.targetUpdate),
	maxStep(stepSize),
	entropyCoef(iOption.entropyCoef),
	rewardNorm(iOption.deviceType),
	maxAveReward(iOption.saveThreshold),
	maxSumReward(iOption.sumSaveThreshold),
	tester(behaviorModel, tEnv, iPolicy, iOption, tLogger)
{
	dataSize = 1;
	for (int i = 1; i < dqnOption.inputShape.size(); i ++) {
		dataSize *= dqnOption.inputShape[i];
	}
	LOG4CXX_INFO(logger, "dataSize = " << dataSize);

	for (int i = 0; i < dqnOption.hiddenNums.size(); i ++) {
		testHiddenStates.push_back(torch::zeros({
			dqnOption.hidenLayerNums[i], dqnOption.testBatch, dqnOption.hiddenNums[i]
		}).to(dqnOption.deviceType));
	}
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CNRNN<NetType, EnvType, PolicyType, OptimizerType>::test(const int batchSize, const int epochNum) {
	testHiddenStates = tester.testACRNN(testHiddenStates);
}


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CNRNN<NetType, EnvType, PolicyType, OptimizerType>::train(const int epNum) {
	LOG4CXX_INFO(logger, "----------------------> batch");
	load();

	int step = 0;
	int epCount = 0;
	int roundCount = 0;


	std::vector<int64_t> stepInputShape;
	std::vector<int64_t> batchInputShape;
//	if (dqnOption.isAtari) {
		stepInputShape.push_back(1);
		stepInputShape.push_back(dqnOption.envNum);
		batchInputShape.push_back(maxStep);
		batchInputShape.push_back(dqnOption.envNum);
		int dataNum = 1;
		for (int i = 1; i < inputShape.size(); i ++) {
			dataNum *= inputShape[i];
		}
		stepInputShape.push_back(dataNum);
		batchInputShape.push_back(dataNum);
		LOG4CXX_INFO(logger, "Get step input shape: " << stepInputShape);
//	} else {
//		stepInputShape.push_back(1);
//		stepInputShape.push_back(batchSize);
//
//		batchInputShape.push_back(maxStep);
//		for (int i = 0; i < inputShape.size(); i ++) {
//			batchInputShape.push_back(inputShape[i]);
//		}
//	}

	std::vector<torch::Tensor> stepStates;
	for (int i = 0; i < dqnOption.hiddenNums.size(); i ++) {
		stepStates.push_back(torch::zeros({
			dqnOption.hidenLayerNums[i], dqnOption.envNum, dqnOption.hiddenNums[i]
		}).to(dqnOption.deviceType));
	}

//	std::vector<std::vector<float>> statesVec;
//	std::vector<std::vector<float>> rewardsVec;
//	std::vector<std::vector<float>> donesVec;
//	std::vector<std::vector<long>> actionsVec;

	std::vector<float> statRewards(dqnOption.envNum, 0);
	std::vector<float> statLens(dqnOption.envNum, 0);
	std::vector<int> liveCounts(dqnOption.envNum, 0);
	std::vector<float> sumRewards(dqnOption.envNum, 0);
	std::vector<float> sumLens(dqnOption.envNum, 0);

	std::vector<Episode> epBatch(dqnOption.envNum);

	std::vector<float> stateVec = env.reset();
	while (updateNum < epNum) {
//		statesVec.clear();
//		rewardsVec.clear();
//		donesVec.clear();
//		actionsVec.clear();
		torch::Tensor lastValueTensor;
		{
			torch::NoGradGuard guard;

//		for (step = 0; step < maxStep; step ++) {
		while (eps.size() < dqnOption.batchSize) { //TODO: replace batchSize with envNum in almost all cases
			updateNum ++;

			torch::Tensor stateTensor = torch::from_blob(stateVec.data(), stepInputShape).div(dqnOption.inputScale).to(deviceType);
//			LOG4CXX_INFO(logger, "stateTensor in step: " << stateTensor.max() << stateTensor.mean());
			std::vector<torch::Tensor> rc = bModel.forward(stateTensor, stepStates);
			auto actionProbs =  torch::softmax(rc[0], -1);
			std::vector<int64_t> actions = policy.getActions(actionProbs);
//			LOG4CXX_INFO(logger, "actions: " << actions);
//			LOG4CXX_INFO(logger, "state: " << stateTensor);


			auto stepResult = env.step(actions, false);
			auto nextStateVec = std::get<0>(stepResult);
			auto rewardVec = std::get<1>(stepResult);
			auto doneVec = std::get<2>(stepResult);

			const int bulkSize = stateVec.size() / dqnOption.envNum;
			for (int k = 0; k < dqnOption.envNum; k ++) {
				epBatch[k].actions.push_back(actions[k]);
				epBatch[k].rewards.push_back(rewardVec[k]);
				//TODO: It's a copy or ref? Would the data destructed? Suppose copied
				epBatch[k].states.insert(epBatch[k].states.end(), stateVec.begin() + bulkSize * k, stateVec.begin() + bulkSize * (k + 1));
			}

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

					statLens[i] = 0;
					statRewards[i] = 0;

					////////////////////////////RNN
					eps.push(epBatch[i]);
					epBatch[i] = Episode();

					if (dqnOption.multiLifes) {
						liveCounts[i] ++;
						if (liveCounts[i] >= dqnOption.donePerEp) {
							roundCount ++;
							LOG4CXX_INFO(logger, "Wrapper episode " << i << " ----------------------------> " << sumRewards[i]);
							tLogger.add_scalar("train/sumLen", roundCount, sumLens[i]);
							tLogger.add_scalar("train/sumReward", roundCount, sumRewards[i]);

							liveCounts[i] = 0;
							sumRewards[i] = 0;
							sumLens[i] = 0;
						}
					}
				}
			}

//			statesVec.push_back(stateVec);
//			rewardsVec.push_back(rewardVec);
//			donesVec.push_back(doneMaskVec);
//			actionsVec.push_back(actions);

			stateVec = nextStateVec;
		}
//		torch::Tensor lastStateTensor = torch::from_blob(stateVec.data(), stepInputShape).div(dqnOption.inputScale).to(deviceType);
//		LOG4CXX_INFO(logger, "stateTensor in last: " << lastStateTensor.max() << lastStateTensor.mean());

//		auto rc = bModel.forward(lastStateTensor, stepStates);
//		lastValueTensor = rc[1].squeeze(-1).detach();
//		LOG4CXX_INFO(logger, "lastValue: " << lastValueTensor);
		}

////////////////////////////////////////////////////////// INPUT ///////////////////////////////////////////////////////
		std::vector<Episode> epsData;
		for (int i = 0; i < dqnOption.batchSize; i ++) {
			epsData.push_back(eps.front());
			eps.pop();
		}
		std::sort(epsData.begin(), epsData.end(),
				[](const Episode& a, const Episode& b) -> bool {
			return a.states.size() > b.states.size();
		});

		std::vector<torch::Tensor> returnVec;
		std::vector<torch::Tensor> actionVec;
		for (int i = 0; i < epsData.size(); i ++) {
			long seqLen = epsData[i].actions.size();

			epsData[i].returns = std::vector<float>(epsData[i].actions.size(), 0);
			epsData[i].returns[epsData[i].returns.size() - 1] = epsData[i].rewards[epsData[i].rewards.size() - 1];
			for (int j = epsData[i].returns.size() - 2; j >= 0; j --) {
				epsData[i].returns[j] = epsData[i].rewards[j] + epsData[i].returns[j + 1] * dqnOption.gamma;
			}
			torch::Tensor returnTensor = torch::from_blob(epsData[i].returns.data(), {seqLen});
			returnVec.push_back(returnTensor);

			torch::Tensor actionTensor = torch::from_blob(epsData[i].actions.data(), {seqLen, 1}, longOpt);
			actionVec.push_back(actionTensor);
		}
		torch::Tensor returnTensor = torch::cat(returnVec, 0).to(dqnOption.deviceType);
		torch::Tensor actionTensor = torch::cat(actionVec, 0).to(dqnOption.deviceType);

		std::vector<long> seqLens;
		std::vector<torch::Tensor> inputStateVec;
		for (int i = 0; i < dqnOption.batchSize; i ++) {
			torch::Tensor stateTensor = torch::from_blob(epsData[i].states.data(), {(long)epsData[i].actions.size(), 4});
			inputStateVec.push_back(stateTensor);
			seqLens.push_back(epsData[i].actions.size());
		}
		torch::Tensor stateTensor = torch::cat(inputStateVec, 0).to(dqnOption.deviceType); //{sum(seqLen), bulkSize}
//		torch::Tensor stateTensor = torch::nn::utils::rnn::pad_sequence(inputStateVec);
//		stateTensor = stateTensor.to(deviceType);
//		torch::Tensor seqTensor = torch::from_blob(seqLens.data(), {dqnOption.batchSize}, longOpt);
//		seqTensor = seqTensor.to(deviceType);


		std::vector<torch::Tensor> trainStates;
		for (int i = 0; i < dqnOption.hiddenNums.size(); i ++) {
			trainStates.push_back(torch::zeros({
				dqnOption.hidenLayerNums[i], dqnOption.batchSize, dqnOption.hiddenNums[i]
			}).to(dqnOption.deviceType));
		}

		auto output = bModel.forward(stateTensor, seqLens, trainStates);
		const int maxSeqLen = seqLens[0];


////////////////////////////////////////////////////// OUTPUT ////////////////////////////////////////////////////////
		auto actionOutputTensor = output[0]; //{sum(seqLen), actionNum}
		torch::Tensor valueTensor = output[1]; //{sum(seqLen)}

		auto advTensor = (returnTensor - valueTensor).detach();
		torch::Tensor valueLoss = torch::nn::functional::mse_loss(valueTensor, returnTensor);

		auto actionProbTensor = torch::softmax(actionOutputTensor, -1); //{maxstep, batch, actionNum}
		auto actionLogTensor = torch::log_softmax(actionOutputTensor, -1); //{maxStep, batch, actionNum}
		torch::Tensor entropyLoss = - (actionProbTensor * actionLogTensor).sum(-1).mean();


		auto actPiTensor = actionLogTensor.gather(-1, actionTensor).squeeze(-1);
		torch::Tensor actLoss = - (actPiTensor * advTensor.detach()).mean();


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
			tLogger.add_scalar("loss/aLoss", updateNum, aLossV);
			tLogger.add_scalar("loss/vLoss", updateNum, vLossV);
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
void A2CNRNN<NetType, EnvType, PolicyType, OptimizerType>::save() {
	if (!dqnOption.saveModel) {
		return;
	}

	AlgUtils::SaveModel(bModel, optimizer, dqnOption.savePathPrefix, logger);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CNRNN<NetType, EnvType, PolicyType, OptimizerType>::saveByReward(float reward) {
	if (!dqnOption.saveModel) {
		return;
	}

	std::string path = dqnOption.savePathPrefix + "_" + std::to_string(reward);
	AlgUtils::SaveModel(bModel, optimizer, path, logger);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CNRNN<NetType, EnvType, PolicyType, OptimizerType>::load() {
	if (!dqnOption.loadModel) {
		return;
	}

	AlgUtils::LoadModel(bModel, optimizer, dqnOption.loadOptimizer, dqnOption.loadPathPrefix, logger);
}



#endif /* INC_ALG_A2CNRNN_HPP_ */
