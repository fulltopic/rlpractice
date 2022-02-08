/*
 * a2cgrutrunc.hpp
 *
 *  Created on: Feb 7, 2022
 *      Author: zf
 */

#ifndef INC_ALG_RNN_A2CGRUTRUNC_HPP_
#define INC_ALG_RNN_A2CGRUTRUNC_HPP_



#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <tensorboard_logger.h>

#include <vector>
#include <queue>
#include <algorithm>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/stats.h"
#include "gymtest/utils/inputnorm.h"
#include "alg/utils/dqnoption.h"
#include "alg/utils/algrnntester.hpp"
#include "alg/utils/utils.hpp"


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
class A2CGRUTrunc {
public:
	struct Episode {
		std::vector<float> states;
		std::vector<int64_t> actions;
		std::vector<float> rewards;
		std::vector<float> returns;

		std::vector<torch::Tensor> startState; //TODO: copy
		float lastValue = 0;
		int seqLen = 0;
	};

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

	uint32_t updateNum = 0;
	uint32_t testEpCount = 0;

	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("a2cgrutrunc");
	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	torch::TensorOptions devLongOpt;

	InputNorm rewardNorm;

	float maxAveReward;
	float maxSumReward;

	AlgRNNTester<NetType, EnvType, PolicyType> tester;
	std::vector<torch::Tensor> testHiddenStates;
	std::queue<Episode> eps;
	std::vector<int64_t> stepFcInputShape;
	std::vector<int64_t> stepConvInputShape;
	std::vector<int64_t> batchFcInputShape;
	std::vector<int64_t> batchConvInputShape;


	void save();
	void saveByReward(float reward);

	void trainBatch(const int epNum); //batched

public:
	A2CGRUTrunc(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, int stepSize, DqnOption option);
	~A2CGRUTrunc() = default;
	A2CGRUTrunc(const A2CGRUTrunc& ) = delete;

	void train(const int epNum); //batched
	void test();
	void load();
};

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
A2CGRUTrunc<NetType, EnvType, PolicyType, OptimizerType>::A2CGRUTrunc(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer,
		int stepSize, const DqnOption iOption):
	bModel(behaviorModel),
	env(iEnv),
	testEnv(tEnv),
	policy(iPolicy),
	optimizer(iOptimizer),
	dqnOption(iOption),
	deviceType(iOption.deviceType),
	inputShape(iOption.inputShape),
	tLogger(iOption.tensorboardLogPath.c_str()),
	rewardNorm(iOption.deviceType),
	maxAveReward(iOption.saveThreshold),
	maxSumReward(iOption.sumSaveThreshold),
	tester(behaviorModel, tEnv, iPolicy, iOption, tLogger)
{
	for (int i = 0; i < dqnOption.hiddenNums.size(); i ++) {
		testHiddenStates.push_back(torch::zeros({
			dqnOption.hidenLayerNums[i], dqnOption.testBatch, dqnOption.hiddenNums[i]
		}).to(dqnOption.deviceType));
	}

	devLongOpt = torch::TensorOptions().dtype(torch::kLong).device(dqnOption.deviceType);

	stepFcInputShape.push_back(dqnOption.envNum);
	stepFcInputShape.push_back(1); //TODO: tmp. To manage them by cellNum, layerNum etc
	batchFcInputShape.push_back(dqnOption.batchSize); //TODO: Check the sequence
	batchFcInputShape.push_back(dqnOption.maxStep);
	int dataNum = 1;
	for (int i = 0; i < inputShape.size(); i ++) {
		dataNum *= inputShape[i];
	}
	stepFcInputShape.push_back(dataNum);
	batchFcInputShape.push_back(dataNum);

	stepConvInputShape.push_back(dqnOption.envNum);
	stepConvInputShape.push_back(1);
	batchConvInputShape.push_back(dqnOption.batchSize);
	batchConvInputShape.push_back(dqnOption.maxStep); //TODO: May be viewed as single sequence
	for (auto& inputDim: inputShape) {
		stepConvInputShape.push_back(inputDim);
		batchConvInputShape.push_back(inputDim);
	}

//	LOG4CXX_INFO(logger, "Get step input shape: " << stepInputShape);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CGRUTrunc<NetType, EnvType, PolicyType, OptimizerType>::test() {
	tester.testAC();
}


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CGRUTrunc<NetType, EnvType, PolicyType, OptimizerType>::train(const int epNum) {
	LOG4CXX_INFO(logger, "----------------------> training");
	load();

	int step = 0;
	int epCount = 0;
	int roundCount = 0;

	std::vector<float> statRewards(dqnOption.envNum, 0);
	std::vector<float> statLens(dqnOption.envNum, 0);
	std::vector<int> liveCounts(dqnOption.envNum, 0);
	std::vector<float> sumRewards(dqnOption.envNum, 0);
	std::vector<float> sumLens(dqnOption.envNum, 0);


	std::vector<Episode> epBatch(dqnOption.envNum);
	std::vector<torch::Tensor> stepStates = bModel.createHStates(dqnOption.envNum, dqnOption.deviceType);
	for (int i = 0; i < dqnOption.envNum; i ++) {
		for (int j = 0; j < dqnOption.gruCellNum; j ++) {
			//stepState: {layerNum, envNum, other}
			//startStates may be GPU type
			torch::Tensor hState = stepStates[j].select(1, i).clone().detach();
			epBatch[i].startState.push_back(hState);
//			LOG4CXX_INFO(logger, "start state for " << i << ": " << hState.sizes());
//			epBatch[i].startState.push_back(stepStates[j][i].clone().detach());
		}
	}
	std::vector<float> stateVec = env.reset();
	std::vector<bool> doneVec(dqnOption.envNum, false);
	while (step < epNum) {
		{
			torch::NoGradGuard guard;

			while (eps.size() < dqnOption.batchSize) { //TODO: replace batchSize with envNum in almost all cases
				step ++;

				torch::Tensor stateTensor = torch::from_blob(stateVec.data(), stepConvInputShape).div(dqnOption.inputScale).to(deviceType);
//				stateTensor = stateTensor.narrow(2, 0, 3);
//				LOG4CXX_INFO(logger, "stateTensor in step: " << stateTensor.max() << stateTensor.mean());
				std::vector<torch::Tensor> rc = bModel.forward(stateTensor, stepStates);
				torch::Tensor rcValue = rc[1];
				auto actionProbs =  torch::softmax(rc[0], -1);
//				LOG4CXX_INFO(logger, "actionProbs " << actionProbs.sizes());
				std::vector<int64_t> actions = policy.getActions(actionProbs);
//				LOG4CXX_INFO(logger, "actions: " << actions);
//				LOG4CXX_INFO(logger, "state: " << stateTensor);

				for (int k = 0; k < dqnOption.envNum; k ++) {
					if ((epBatch[k].seqLen >= dqnOption.maxStep) || doneVec[k]) {
						if (doneVec[k]) {
							epBatch[k].lastValue = 0;
						} else {
							epBatch[k].lastValue = rcValue[k].item<float>();
						}
						eps.push(epBatch[k]);

						epBatch[k] = Episode();

						if (doneVec[k]) {
							bModel.resetHState(k, stepStates);
						}
						for (int gruIndex = 0; gruIndex < dqnOption.hidenLayerNums.size(); gruIndex ++) {
							//stepState: {envNum, 1(step), other}
							//startStates may be GPU type
							torch::Tensor hState = stepStates[gruIndex].select(1, k).clone().detach();
							epBatch[k].startState.push_back(hState);
//							LOG4CXX_INFO(logger, "start state push " << k << ": " << hState.sizes());
						}
					}
				}


				auto stepResult = env.step(actions, false);
				auto nextStateVec = std::get<0>(stepResult);
				auto rewardVec = std::get<1>(stepResult);
				doneVec = std::get<2>(stepResult);

				const int bulkSize = stateVec.size() / dqnOption.envNum;
				for (int k = 0; k < dqnOption.envNum; k ++) {
					epBatch[k].actions.push_back(actions[k]);
					epBatch[k].rewards.push_back(rewardVec[k]);
					epBatch[k].states.insert(epBatch[k].states.end(), stateVec.begin() + bulkSize * k, stateVec.begin() + bulkSize * (k + 1));

					epBatch[k].seqLen ++;
				}

				Stats::UpdateReward(statRewards, rewardVec);
				Stats::UpdateLen(statLens);

//				LOG4CXX_INFO(logger, "rewardVec" << step << ": " << rewardVec);
				for (int i = 0; i < doneVec.size(); i ++) {
//					LOG4CXX_INFO(logger, "dones: " << i << ": " << doneVec[i]);
					if (doneVec[i]) {
						epCount ++;

						sumRewards[i] += statRewards[i];
						sumLens[i] += statLens[i];

						tLogger.add_scalar("train/len", epCount, statLens[i]);
						tLogger.add_scalar("train/reward", epCount, statRewards[i]);
						LOG4CXX_INFO(logger, "ep " << step << ": " << statLens[i] << ", " << statRewards[i]);

						statLens[i] = 0;
						statRewards[i] = 0;

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
				stateVec = nextStateVec;
			}

			torch::Tensor stateTensor = torch::from_blob(stateVec.data(), stepConvInputShape).div(dqnOption.inputScale).to(deviceType);
		}

////////////////////////////////////////////////////////// INPUT ///////////////////////////////////////////////////////
		updateNum ++;

		std::vector<Episode> epsData;
		for (int i = 0; i < dqnOption.batchSize; i ++) {
			epsData.push_back(eps.front());
			eps.pop();
		}
		std::sort(epsData.begin(), epsData.end(),
				[](const Episode& a, const Episode& b) -> bool {
			return a.seqLen > b.seqLen;
		});

		std::vector<torch::Tensor> returnVec;
		std::vector<torch::Tensor> actionVec;
		std::vector<std::vector<torch::Tensor>> hiddenStateVec(dqnOption.gruCellNum);
		for (int i = 0; i < epsData.size(); i ++) {
			long seqLen = epsData[i].seqLen;
//			LOG4CXX_INFO(logger, "sorted seqLen = " << seqLen);

			epsData[i].returns = std::vector<float>(seqLen, 0);
			float returnValue = epsData[i].lastValue;
//			epsData[i].returns[seqLen - 1] = epsData[i].rewards[seqLen - 1];
			for (int j = epsData[i].returns.size() - 1; j >= 0; j --) {
				returnValue = returnValue * dqnOption.gamma + epsData[i].rewards[j];
				epsData[i].returns[j] = returnValue;
			}
			torch::Tensor returnTensor = torch::from_blob(epsData[i].returns.data(), {seqLen, 1});
			returnVec.push_back(returnTensor);
//			LOG4CXX_INFO(logger, "return Tensor " << returnTensor);

			torch::Tensor actionTensor = torch::from_blob(epsData[i].actions.data(), {seqLen, 1}, longOpt);
			actionVec.push_back(actionTensor);

			for (int gruIndex = 0; gruIndex < dqnOption.hidenLayerNums.size(); gruIndex ++) {
				hiddenStateVec[gruIndex].push_back(epsData[i].startState[gruIndex]);
			}
//			LOG4CXX_INFO(logger, "actionTensor " << actionTensor);
		}
		torch::Tensor returnTensor = torch::cat(returnVec, 0).to(dqnOption.deviceType);
		torch::Tensor actionTensor = torch::cat(actionVec, 0).to(dqnOption.deviceType);
		std::vector<torch::Tensor> hiddenState;
		for (int gruIndex = 0; gruIndex < dqnOption.hidenLayerNums.size(); gruIndex ++) {
			hiddenState.push_back(torch::stack(hiddenStateVec[gruIndex], 1));
		}
//		LOG4CXX_INFO(logger, "sum returnTensor " << returnTensor);
//		LOG4CXX_INFO(logger, "sum actionTensor " << actionTensor);

		std::vector<long> seqLens;
		std::vector<torch::Tensor> inputStateVec;
		for (int i = 0; i < dqnOption.batchSize; i ++) {
			int seqLen = epsData[i].seqLen;
			seqLens.push_back(seqLen);
//			LOG4CXX_INFO(logger, "state seqLen " << seqLen);

			torch::Tensor stateTensor = torch::from_blob(epsData[i].states.data(), {(long)seqLen, 4}); //TODO: 4 tmp
			inputStateVec.push_back(stateTensor);
//			LOG4CXX_INFO(logger, "stateTensor " << stateTensor);
		}
		torch::Tensor stateTensor = torch::cat(inputStateVec, 0).to(dqnOption.deviceType); //{sum(seqLen), bulkSize}
//		stateTensor = stateTensor.narrow(1, 0, 3);
//		LOG4CXX_INFO(logger, "sum stateTensor " << stateTensor);

		//TODO: hiddenState
		auto output = bModel.forward(stateTensor, seqLens, hiddenState);
//		auto output = bModel.forward(stateTensor, shuffleIndex, seqLens);
//		const int maxSeqLen = seqLens[0];


////////////////////////////////////////////////////// OUTPUT ////////////////////////////////////////////////////////
		auto actionOutputTensor = output[0]; //{sum(seqLen), actionNum}
		torch::Tensor valueTensor = output[1]; //{sum(seqLen)}
//		LOG4CXX_INFO(logger, "actionOutput " << actionOutputTensor);
//		LOG4CXX_INFO(logger, "valueOutput " << valueTensor);

		auto advTensor = (returnTensor - valueTensor).detach();
		torch::Tensor valueLoss = torch::nn::functional::mse_loss(valueTensor, returnTensor);
//		LOG4CXX_INFO(logger, "advTensor " << advTensor);

		auto actionProbTensor = torch::softmax(actionOutputTensor, -1); //{maxstep, batch, actionNum}
		auto actionLogTensor = torch::log_softmax(actionOutputTensor, -1); //{maxStep, batch, actionNum}
		torch::Tensor entropyLoss = - (actionProbTensor * actionLogTensor).sum(-1).mean();
//		LOG4CXX_INFO(logger, "actionProb " << actionProbTensor);
//		LOG4CXX_INFO(logger, "actionLogProb " << actionLogTensor);


		auto actPiTensor = actionLogTensor.gather(-1, actionTensor).squeeze(-1);
		torch::Tensor actLoss = - (actPiTensor * advTensor.detach()).mean();
//		LOG4CXX_INFO(logger, "actPiTensor " << actPiTensor);


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
			test();
		}
		}
	}

	save();
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CGRUTrunc<NetType, EnvType, PolicyType, OptimizerType>::save() {
	if (!dqnOption.saveModel) {
		return;
	}

	AlgUtils::SaveModel(bModel, optimizer, dqnOption.savePathPrefix, logger);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CGRUTrunc<NetType, EnvType, PolicyType, OptimizerType>::saveByReward(float reward) {
	if (!dqnOption.saveModel) {
		return;
	}

	std::string path = dqnOption.savePathPrefix + "_" + std::to_string(reward);
	AlgUtils::SaveModel(bModel, optimizer, path, logger);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CGRUTrunc<NetType, EnvType, PolicyType, OptimizerType>::load() {
	if (!dqnOption.loadModel) {
		return;
	}

	AlgUtils::LoadModel(bModel, optimizer, dqnOption.loadOptimizer, dqnOption.loadPathPrefix, logger);
}




#endif /* INC_ALG_RNN_A2CGRUTRUNC_HPP_ */
