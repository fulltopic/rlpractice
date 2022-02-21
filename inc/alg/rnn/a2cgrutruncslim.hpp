/*
 * a2cgrutruncslim.hpp
 *
 *  Created on: Feb 13, 2022
 *      Author: zf
 */

#ifndef INC_ALG_RNN_A2CGRUTRUNCSLIM_HPP_
#define INC_ALG_RNN_A2CGRUTRUNCSLIM_HPP_




#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <tensorboard_logger.h>

#include <vector>
#include <queue>
#include <algorithm>
#include <random>
#include <chrono>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/stats.h"
#include "gymtest/utils/inputnorm.h"
#include "alg/utils/dqnoption.h"
#include "alg/utils/algrnntester.hpp"
#include "alg/utils/utils.hpp"


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
class A2CGRUTruncSlim {
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

	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("A2CGRUTruncSlim");
	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	torch::TensorOptions devLongOpt;
	torch::TensorOptions devOpt;

	InputNorm rewardNorm;

	float maxAveReward;
	float maxSumReward;

	AlgRNNTester<NetType, EnvType, PolicyType> tester;
//	std::vector<torch::Tensor> testHiddenStates;
	std::queue<Episode> eps;
	std::vector<int64_t> stepFcInputShape;
	std::vector<int64_t> stepConvInputShape;
	std::vector<int64_t> batchFcInputShape;
	std::vector<int64_t> batchConvInputShape;


	void save();
	void saveByReward(float reward);

	void trainBatch(const int epNum); //batched

public:
	A2CGRUTruncSlim(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, DqnOption option);
	~A2CGRUTruncSlim() = default;
	A2CGRUTruncSlim(const A2CGRUTruncSlim& ) = delete;

	void train(const int epNum); //batched
	void test();
	void load();
};

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
A2CGRUTruncSlim<NetType, EnvType, PolicyType, OptimizerType>::A2CGRUTruncSlim(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer,
		const DqnOption iOption):
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
	devLongOpt = torch::TensorOptions().dtype(torch::kLong).device(dqnOption.deviceType);
	devOpt = torch::TensorOptions().device(dqnOption.deviceType);

//	for (int i = 0; i < dqnOption.hiddenNums.size(); i ++) {
//		testHiddenStates.push_back(torch::zeros({
//			dqnOption.hidenLayerNums[i], dqnOption.testBatch, dqnOption.hiddenNums[i]
//		}).to(dqnOption.deviceType));
//	}

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
void A2CGRUTruncSlim<NetType, EnvType, PolicyType, OptimizerType>::test() {
	tester.testAC();
}


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CGRUTruncSlim<NetType, EnvType, PolicyType, OptimizerType>::train(const int epNum) {
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
	std::vector<Episode> epBatchNew(dqnOption.envNum);
	std::vector<torch::Tensor> stepStates = bModel.createHStates(dqnOption.envNum, dqnOption.deviceType);
	for (int i = 0; i < dqnOption.envNum; i ++) {
		epBatch[i].startState = bModel.getHState(i, stepStates);
	}
	std::vector<float> stateVec = env.reset();
	std::vector<bool> doneVec(dqnOption.envNum, false);
	while (step < epNum)
	{
		{
			torch::NoGradGuard guard;

			while (eps.size() < dqnOption.batchSize) {
				step ++;

				for (int k = 0; k < dqnOption.envNum; k ++) {
					if (epBatch[k].seqLen >= dqnOption.maxStep) {
						epBatchNew[k] = Episode();
						epBatchNew[k].startState = bModel.getHState(k, stepStates);
					}
				}

				torch::Tensor stateTensor = torch::from_blob(stateVec.data(), stepConvInputShape).div(dqnOption.inputScale).to(deviceType);
//				stateTensor = stateTensor.narrow(2, 0, 3);
//				LOG4CXX_INFO(logger, "stateTensor in step: " << stateTensor.max() << stateTensor.mean());
				std::vector<torch::Tensor> rc = bModel.forward(stateTensor, stepStates);
				torch::Tensor rcValue = rc[1];
				auto actionProbs = torch::softmax(rc[0], -1);
//				LOG4CXX_INFO(logger, "actionProbs " << actionProbs);
				std::vector<int64_t> actions = policy.getActions(actionProbs);
//				LOG4CXX_INFO(logger, "actions: " << actions);
//				LOG4CXX_INFO(logger, "state: " << stateTensor);

				for (int k = 0; k < dqnOption.envNum; k ++) {
					if (epBatch[k].seqLen >= dqnOption.maxStep) {
						epBatch[k].lastValue = rcValue[k].item<float>();

						eps.push(epBatch[k]);

						epBatch[k] = epBatchNew[k];
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

								bModel.resetHState(i, stepStates);
							}
						} else {
							bModel.resetHState(i, stepStates);
						}

						//RNN
						epBatch[i].lastValue = 0;
						eps.push(epBatch[i]);

						epBatch[i] = Episode();
						epBatch[i].startState = bModel.getHState(i, stepStates);
					}
				}
				stateVec = nextStateVec;
			}
		}

////////////////////////////////////////////////////////// INPUT ///////////////////////////////////////////////////////
		updateNum ++;
//		LOG4CXX_INFO(logger, "updateNum " << updateNum);

		int minSeqLen = dqnOption.maxStep;
		std::vector<Episode> epsData;
		for (int i = 0; i < dqnOption.batchSize; i ++) {
			epsData.push_back(eps.front());
			eps.pop();

			minSeqLen = std::min(minSeqLen, epsData[epsData.size() - 1].seqLen);
		}
//		LOG4CXX_INFO(logger, "minSeq = " << minSeqLen);

//		auto seed = std::chrono::system_clock::now().time_since_epoch().count();
//		std::shuffle(epsData.begin(), epsData.end(), std::default_random_engine(seed));


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
			returnTensor = returnTensor.narrow(0, 0, minSeqLen);
			returnVec.push_back(returnTensor);
//			LOG4CXX_INFO(logger, "return Tensor " << returnTensor);

			torch::Tensor actionTensor = torch::from_blob(epsData[i].actions.data(), {seqLen, 1}, longOpt);
			actionTensor = actionTensor.narrow(0, 0, minSeqLen);
			actionVec.push_back(actionTensor);
//			LOG4CXX_INFO(logger, "actions: " << epsData[i].actions);
//			LOG4CXX_INFO(logger, "actionTensor: " << actionTensor);

			for (int gruIndex = 0; gruIndex < dqnOption.hidenLayerNums.size(); gruIndex ++) {
				hiddenStateVec[gruIndex].push_back(epsData[i].startState[gruIndex]);
			}
//			LOG4CXX_INFO(logger, "actionTensor " << actionTensor);
		}

		//returnVec[0] ~ {seqLen, dim0, dim1, ... , dimn}, returnTensor ~ {seqLen, batch * dim0, dim1...n}
		torch::Tensor returnTensor = torch::cat(returnVec, 0).view({minSeqLen * dqnOption.batchSize}).to(dqnOption.deviceType);
		torch::Tensor actionTensor = torch::cat(actionVec, 0).view({minSeqLen * dqnOption.batchSize, 1}).to(dqnOption.deviceType);
		std::vector<torch::Tensor> hiddenState;
		for (int gruIndex = 0; gruIndex < dqnOption.hidenLayerNums.size(); gruIndex ++) {
			hiddenState.push_back(torch::stack(hiddenStateVec[gruIndex], 1));
		}
//		LOG4CXX_INFO(logger, "sum returnTensor " << returnTensor.sizes());
//		LOG4CXX_INFO(logger, "sum actionTensor " << actionTensor.sizes());

		std::vector<torch::Tensor> inputStateVec;
		for (int i = 0; i < dqnOption.batchSize; i ++) {
			int seqLen = epsData[i].seqLen;
//			LOG4CXX_INFO(logger, "state seqLen " << seqLen);

			std::vector<long> seqShape{seqLen};
			seqShape.insert(seqShape.end(), dqnOption.inputShape.begin(), dqnOption.inputShape.end());
			torch::Tensor stateTensor = torch::from_blob(epsData[i].states.data(), seqShape);
			stateTensor = stateTensor.narrow(0, 0, minSeqLen);
			inputStateVec.push_back(stateTensor);
//			LOG4CXX_INFO(logger, "stateTensor " << stateTensor);
		}
		std::vector<long> seqShape{dqnOption.batchSize * minSeqLen};
		seqShape.insert(seqShape.end(), dqnOption.inputShape.begin(), dqnOption.inputShape.end());
		torch::Tensor stateTensor = torch::cat(inputStateVec, 0).view(seqShape);
		stateTensor = stateTensor.div(dqnOption.inputScale).to(dqnOption.deviceType);

		auto output = bModel.forward(stateTensor, dqnOption.batchSize, minSeqLen, hiddenState, dqnOption.deviceType);


////////////////////////////////////////////////////// OUTPUT ////////////////////////////////////////////////////////
		auto actionOutputTensor = output[0]; //{sum(seqLen), actionNum}
		torch::Tensor valueTensor = output[1].view({dqnOption.batchSize * minSeqLen}); //{sum(seqLen)}
//		LOG4CXX_INFO(logger, "actionOutput " << actionOutputTensor);
//		LOG4CXX_INFO(logger, "valueOutput " << valueTensor);

		auto advTensor = (returnTensor - valueTensor);
		torch::Tensor valueLoss = torch::nn::functional::mse_loss(valueTensor, returnTensor);
//		LOG4CXX_INFO(logger, "advTensor " << advTensor);

		auto actionProbTensor = torch::softmax(actionOutputTensor, -1); //{sum(seqLen), actionNum}
		auto actionLogTensor = torch::log_softmax(actionOutputTensor, -1); //{sum(seqLen), actionNum}
		torch::Tensor entropyLoss = - (actionProbTensor * actionLogTensor).sum(-1).mean();
//		LOG4CXX_INFO(logger, "actionProb " << actionProbTensor);
//		LOG4CXX_INFO(logger, "actionLogProb " << actionLogTensor);


		auto actPiTensor = actionLogTensor.gather(-1, actionTensor).squeeze(-1);
		torch::Tensor actLoss = - (actPiTensor * advTensor.detach()).mean();
//		LOG4CXX_INFO(logger, "advTensor " << advTensor);
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

			tLogger.add_scalar("loss/loss", step, lossV);
			tLogger.add_scalar("loss/aLoss", step, aLossV);
			tLogger.add_scalar("loss/vLoss", step, vLossV);
			tLogger.add_scalar("loss/entropy", step, entropyV);
			tLogger.add_scalar("loss/v", step, valueV);
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
void A2CGRUTruncSlim<NetType, EnvType, PolicyType, OptimizerType>::save() {
	if (!dqnOption.saveModel) {
		return;
	}

	AlgUtils::SaveModel(bModel, optimizer, dqnOption.savePathPrefix, logger);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CGRUTruncSlim<NetType, EnvType, PolicyType, OptimizerType>::saveByReward(float reward) {
	if (!dqnOption.saveModel) {
		return;
	}

	std::string path = dqnOption.savePathPrefix + "_" + std::to_string(reward);
	AlgUtils::SaveModel(bModel, optimizer, path, logger);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CGRUTruncSlim<NetType, EnvType, PolicyType, OptimizerType>::load() {
	if (!dqnOption.loadModel) {
		return;
	}

	AlgUtils::LoadModel(bModel, optimizer, dqnOption.loadOptimizer, dqnOption.loadPathPrefix, logger);
}






#endif /* INC_ALG_RNN_A2CGRUTRUNCSLIM_HPP_ */
