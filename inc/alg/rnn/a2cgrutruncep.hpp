/*
 * a2cgrutruncep.hpp
 *
 *  Created on: Feb 12, 2022
 *      Author: zf
 */

#ifndef INC_ALG_RNN_A2CGRUTRUNCEP_HPP_
#define INC_ALG_RNN_A2CGRUTRUNCEP_HPP_


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
class A2CGRUTruncEpisode {
public:
	struct Episode {
		std::vector<float> states;
		std::vector<int64_t> actions;
		std::vector<float> rewards;

		std::vector<torch::Tensor> startState; //TODO: copy

//		std::vector<float> gae;
		std::vector<float> returnValue;
		int seqLen = 0;

		torch::Tensor stateTensor;
		torch::Tensor actionTensor;
		torch::Tensor returnTensor;
	};

	struct TensorEpisode {
		torch::Tensor state;
		torch::Tensor action;
		torch::Tensor returnValue;
		std::vector<torch::Tensor> startState;
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

	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("A2CGRUTruncEpisode");
	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	torch::TensorOptions devLongOpt;
	torch::TensorOptions devOpt;

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
	A2CGRUTruncEpisode(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, DqnOption option);
	~A2CGRUTruncEpisode() = default;
	A2CGRUTruncEpisode(const A2CGRUTruncEpisode& ) = delete;

	void train(const int epNum); //batched
	void test();
	void load();
};

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
A2CGRUTruncEpisode<NetType, EnvType, PolicyType, OptimizerType>::A2CGRUTruncEpisode(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer,
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

	for (int i = 0; i < dqnOption.hiddenNums.size(); i ++) {
		testHiddenStates.push_back(torch::zeros({
			dqnOption.hidenLayerNums[i], dqnOption.testBatch, dqnOption.hiddenNums[i]
		}).to(dqnOption.deviceType));
	}

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
void A2CGRUTruncEpisode<NetType, EnvType, PolicyType, OptimizerType>::test() {
	tester.testAC();
}


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CGRUTruncEpisode<NetType, EnvType, PolicyType, OptimizerType>::train(const int epNum) {
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
//	for (int i = 0; i < dqnOption.envNum; i ++) {
//		epBatch[i].startState = bModel.getHState(i, stepStates);
//	}
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
							bModel.resetHState(i, stepStates); //TODO
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

		/////TODO: TBC
////////////////////////////////////////////////////////// INPUT ///////////////////////////////////////////////////////
		updateNum ++;
//		LOG4CXX_INFO(logger, "updateNum " << updateNum);

		std::vector<Episode> epsRaw;
		for (int i = 0; i < dqnOption.batchSize; i ++) {
			epsRaw.push_back(eps.front());
			eps.pop();
		}

		//Return values
		for (int i = 0; i < epsRaw.size(); i ++) {
			int seqLen = epsRaw[i].seqLen;

			float rValue = 0;
			epsRaw[i].returnValue = std::vector<float>(seqLen, 0);
			for (int k = seqLen - 1; k >= 0; k --) {
				rValue = epsRaw[i].rewards[k] + dqnOption.gamma * rValue;
				epsRaw[i].returnValue[k] = rValue;
			}
		}

		//tensor
		std::vector<torch::Tensor> trainHState = bModel.createHState(epsRaw.size(), dqnOption.deviceType);
		for (int i = 0; i < epsRaw.size(); i ++) {
			std::vector<long> seqShape{epsRaw[i].seqLen};
			seqShape.insert(seqShape.end(), dqnOption.inputShape.begin(), dqnOption.inputShape.end());

			epsRaw[i].stateTensor = torch::from_blob(epsRaw[i].states.data(), seqShape).div(dqnOption.inputScale).to(dqnOption.deviceType);
			epsRaw[i].actionTensor = torch::from_blob(epsRaw[i].actions.data(), {epsRaw[i].seqLen, 1}, longOpt).to(dqnOption.deviceType);
			epsRaw[i].returnTensor = torch::from_blob(epsRaw[i].returnValue.data(), {epsRaw[i].seqLen}).to(dqnOption.deviceType);
			epsRaw[i].startState = bModel.getHState(i, trainHState);
		}

		std::vector<int> trunkIndex(epsRaw.size(), 0);

		bool todo = true;
		while (todo) {
			//TODO: change in each loop?
			torch::Tensor indexTensor = torch::randperm(epsRaw.size());
			long* indexPtr = indexTensor.data_ptr<long>();

			std::vector<TensorEpisode> epsData;
			for (int i = 0; i < dqnOption.batchSize; i ++) {
				int index = indexPtr[i];
				if (trunkIndex[index] < epsRaw[index].seqLen) {
					TensorEpisode ep;
					ep.seqLen = std::min((epsRaw[index].seqLen - trunkIndex[index]));

					ep.state = epsRaw[index].stateTensor.narrow(0, trunkIndex[index], ep.seqLen);
					ep.action = epsRaw[index].actionTensor.narrow(0, trunkIndex[index], ep.seqLen);
					ep.returnValue = epsRaw[index].returnTensor.narrow(0, trunkIndex[index], ep.seqLen);
					ep.startState = epsRaw[index].startState;

					trunkIndex[index] += ep.seqLen;

					epsData.push_back(ep);
				}
			}

			if (epsData.size() <= 0) {
				break; //No more trunk
			}

			std::sort(epsData.begin(), epsData.end(),
					[](const TensorEpisode& a, const TensorEpisode& b) -> bool {
				return a.seqLen > b.seqLen;
			});

			std::vector<torch::Tensor> stateVec;
			std::vector<torch::Tensor> actionVec;
			std::vector<torch::Tensor> returnVec;
			for (int i = 0; i < epsData.size(); i ++) {
				stateVec.push_back(epsData[i].state);
				actionVec.push_back(epsData[i].action);
				returnVec.push_back(epsData[i].returnValue);
			}


		}



		std::vector<torch::Tensor> returnVec;
		std::vector<torch::Tensor> actionVec;
		std::vector<std::vector<torch::Tensor>> hiddenStateVec(dqnOption.gruCellNum);
		for (int i = 0; i < epsData.size(); i ++) {
			long seqLen = epsData[i].seqLen;
//			LOG4CXX_INFO(logger, "sorted seqLen = " << seqLen);

			torch::Tensor actionTensor = torch::from_blob(epsData[i].actions.data(), {seqLen, 1}, longOpt);
			actionVec.push_back(actionTensor);

			for (int gruIndex = 0; gruIndex < dqnOption.hidenLayerNums.size(); gruIndex ++) {
				hiddenStateVec[gruIndex].push_back(epsData[i].startState[gruIndex]);
			}
//			LOG4CXX_INFO(logger, "actionTensor " << actionTensor);
		}
//		torch::Tensor returnTensor = torch::cat(returnVec, 0).to(dqnOption.deviceType);
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

			std::vector<long> seqShape{seqLen};
			seqShape.insert(seqShape.end(), dqnOption.inputShape.begin(), dqnOption.inputShape.end());
			torch::Tensor stateTensor = torch::from_blob(epsData[i].states.data(), seqShape);
			inputStateVec.push_back(stateTensor);
//			LOG4CXX_INFO(logger, "stateTensor " << stateTensor);
		}
		torch::Tensor stateTensor = torch::cat(inputStateVec, 0).div(dqnOption.inputScale).to(dqnOption.deviceType); //{sum(seqLen), bulkSize}
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

		/////////////////////////////////////// GAE
		std::vector<float> gaeValues(valueTensor.sizes()[0], 0);
		int gaeIndex = valueTensor.sizes()[0] - 1;
		for (int seqIndex = seqLens.size() - 1; seqIndex >= 0; seqIndex --) {
			int seqLen = seqLens[seqIndex];
			float qValue = epsData[seqIndex].lastValue;
			float gaeValue = 0;
			for (int k = seqLen - 1; k >= 0; k --) {
				float delta = epsData[seqIndex].rewards[k] + dqnOption.gamma * qValue - valueTensor[gaeIndex].item<float>();
				gaeValue = delta + dqnOption.ppoLambda * dqnOption.gamma * gaeValue;
				gaeValues[gaeIndex] = gaeValue;

				qValue = valueTensor[gaeIndex].item<float>();
				gaeIndex --;
			}
		}
		torch::Tensor gaeTensor = (torch::from_blob(gaeValues.data(), valueTensor.sizes()).to(deviceType)).detach();
		torch::Tensor returnTensor = (gaeTensor + valueTensor).detach();
		///////////////////////////////////////GAE End

//		auto advTensor = (returnTensor - valueTensor);
		torch::Tensor valueLoss = torch::nn::functional::mse_loss(valueTensor, returnTensor);
//		LOG4CXX_INFO(logger, "advTensor " << advTensor);

		auto actionProbTensor = torch::softmax(actionOutputTensor, -1); //{maxstep, batch, actionNum}
		auto actionLogTensor = torch::log_softmax(actionOutputTensor, -1); //{maxStep, batch, actionNum}
		torch::Tensor entropyLoss = - (actionProbTensor * actionLogTensor).sum(-1).mean();
//		LOG4CXX_INFO(logger, "actionProb " << actionProbTensor);
//		LOG4CXX_INFO(logger, "actionLogProb " << actionLogTensor);


		auto actPiTensor = actionLogTensor.gather(-1, actionTensor).squeeze(-1);
		torch::Tensor actLoss = - (actPiTensor * gaeTensor).mean();
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
void A2CGRUTruncEpisode<NetType, EnvType, PolicyType, OptimizerType>::save() {
	if (!dqnOption.saveModel) {
		return;
	}

	AlgUtils::SaveModel(bModel, optimizer, dqnOption.savePathPrefix, logger);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CGRUTruncEpisode<NetType, EnvType, PolicyType, OptimizerType>::saveByReward(float reward) {
	if (!dqnOption.saveModel) {
		return;
	}

	std::string path = dqnOption.savePathPrefix + "_" + std::to_string(reward);
	AlgUtils::SaveModel(bModel, optimizer, path, logger);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2CGRUTruncEpisode<NetType, EnvType, PolicyType, OptimizerType>::load() {
	if (!dqnOption.loadModel) {
		return;
	}

	AlgUtils::LoadModel(bModel, optimizer, dqnOption.loadOptimizer, dqnOption.loadPathPrefix, logger);
}


#endif /* INC_ALG_RNN_A2CGRUTRUNCEP_HPP_ */
