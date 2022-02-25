/*
 * ppogrutruncslimgae.hpp
 *
 *  Created on: Feb 22, 2022
 *      Author: zf
 */

#ifndef INC_ALG_RNN_PPOGRUTRUNCSLIMGAE_HPP_
#define INC_ALG_RNN_PPOGRUTRUNCSLIMGAE_HPP_



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
class PPOGRUTruncSlimGae {
public:
	struct Episode {
		std::vector<float> states;
		std::vector<int64_t> actions;
		std::vector<float> rewards;
		std::vector<float> values;
		std::vector<float> pis;

		std::vector<torch::Tensor> startState; //TODO: copy
		float lastValue = 0;
		int seqLen = 0;

		std::vector<float> returns;
		std::vector<float> gaes;
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
	int roundNum = 0;
	int bulkSize = 1;

	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("PPOGRUTruncSlimGae");
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
	PPOGRUTruncSlimGae(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, DqnOption option);
	~PPOGRUTruncSlimGae() = default;
	PPOGRUTruncSlimGae(const PPOGRUTruncSlimGae& ) = delete;

	void train(const int epNum); //batched
	void test();
	void load();
};

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
PPOGRUTruncSlimGae<NetType, EnvType, PolicyType, OptimizerType>::PPOGRUTruncSlimGae(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer,
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
	assert((dqnOption.trajStepNum / dqnOption.batchSize) > 0);
	assert((dqnOption.trajStepNum % dqnOption.batchSize) == 0);
	roundNum = dqnOption.trajStepNum / dqnOption.batchSize;

	devLongOpt = torch::TensorOptions().dtype(torch::kLong).device(dqnOption.deviceType);
	devOpt = torch::TensorOptions().device(dqnOption.deviceType);

	stepFcInputShape.push_back(dqnOption.envNum);
	stepFcInputShape.push_back(1); //TODO: tmp. To manage them by cellNum, layerNum etc
	batchFcInputShape.push_back(dqnOption.batchSize); //TODO: Check the sequence
	batchFcInputShape.push_back(dqnOption.maxStep);
	bulkSize = 1;
	for (int i = 0; i < inputShape.size(); i ++) {
		bulkSize *= inputShape[i];
	}
	stepFcInputShape.push_back(bulkSize);
	batchFcInputShape.push_back(bulkSize);

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
void PPOGRUTruncSlimGae<NetType, EnvType, PolicyType, OptimizerType>::test() {
	tester.testAC();
}


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PPOGRUTruncSlimGae<NetType, EnvType, PolicyType, OptimizerType>::train(const int epNum) {
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
//	std::vector<Episode> epBatchNew(dqnOption.envNum);
	std::vector<torch::Tensor> stepStates = bModel.createHStates(dqnOption.envNum, dqnOption.deviceType);
	for (int i = 0; i < dqnOption.envNum; i ++) {
		epBatch[i].startState = bModel.getHState(i, stepStates);
	}
	std::vector<float> stateVec = env.reset();
//	std::vector<bool> doneVec(dqnOption.envNum, false);
	while (step < epNum)
	{
		{
			torch::NoGradGuard guard;

			while (eps.size() < dqnOption.trajStepNum) {
				step ++;


				torch::Tensor stateTensor = torch::from_blob(stateVec.data(), stepConvInputShape).div(dqnOption.inputScale).to(deviceType);
//				LOG4CXX_INFO(logger, "stateTensor in step: " << stateTensor.max() << stateTensor.mean());
				auto rc = bModel.forwardNext(stateTensor, stepStates);
				torch::Tensor rcValue = std::get<0>(rc)[1];
				torch::Tensor rcAction = std::get<0>(rc)[0];
				std::vector<torch::Tensor> nextStepStates = std::get<1>(rc);

				auto actionProbs = torch::softmax(rcAction, -1);
//				LOG4CXX_INFO(logger, "actionProbs " << actionProbs);
				std::vector<int64_t> actions = policy.getActions(actionProbs);
//				LOG4CXX_INFO(logger, "actions: " << actions);
//				LOG4CXX_INFO(logger, "state: " << stateTensor);

				for (int k = 0; k < dqnOption.envNum; k ++) {
					if (epBatch[k].seqLen >= dqnOption.maxStep) {
						epBatch[k].lastValue = rcValue[k].item<float>();
						eps.push(epBatch[k]);

						epBatch[k] = Episode();
						epBatch[k].startState = bModel.getHState(k, stepStates);
					}
				}

				auto stepResult = env.step(actions, false);
				auto nextStateVec = std::get<0>(stepResult);
				auto rewardVec = std::get<1>(stepResult);
				auto doneVec = std::get<2>(stepResult);

//				const int bulkSize = stateVec.size() / dqnOption.envNum;
				for (int k = 0; k < dqnOption.envNum; k ++) {
					epBatch[k].actions.push_back(actions[k]);
					epBatch[k].rewards.push_back(rewardVec[k]);
					epBatch[k].values.push_back(rcValue[k].item<float>());
					epBatch[k].pis.push_back(actionProbs[k][actions[k]].item<float>());
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

								bModel.resetHState(i, nextStepStates);
							}
						} else {
							bModel.resetHState(i, nextStepStates);
						}

						//RNN
						epBatch[i].lastValue = 0;
						eps.push(epBatch[i]);

						epBatch[i] = Episode();
						epBatch[i].startState = bModel.getHState(i, nextStepStates);
					}
//					else if (epBatch[i].seqLen >= dqnOption.maxStep) {
//						epBatch[i].lastValue =
//					}
				}


				stateVec = nextStateVec;
				stepStates = nextStepStates;
			}
		}

////////////////////////////////////////////////////////// INPUT ///////////////////////////////////////////////////////
//		LOG4CXX_INFO(logger, "updateNum " << updateNum);
		std::vector<Episode> epsData;
		for (int i = 0; i < dqnOption.trajStepNum; i ++) {
			epsData.push_back(eps.front());
			eps.pop();
		}

		std::vector<long> seqLens;
		std::vector<torch::Tensor> returnVec;
		std::vector<torch::Tensor> actionVec;
		std::vector<torch::Tensor> gaeVec;
		std::vector<torch::Tensor> oldPiVec;
		std::vector<torch::Tensor> inputStateVec;
		std::vector<std::vector<torch::Tensor>> hiddenStateVec(dqnOption.gruCellNum);
		for (int i = 0; i < epsData.size(); i ++) {
			long seqLen = epsData[i].seqLen;
			seqLens.push_back(seqLen);
//			LOG4CXX_INFO(logger, "sorted seqLen = " << seqLen);

			float qValue = epsData[i].lastValue;
			float gaeValue = 0;
			epsData[i].returns = std::vector<float>(seqLen, 0);
			epsData[i].gaes = std::vector<float>(seqLen, 0);
			for (int k = seqLen - 1; k >= 0; k --) {
				float delta = epsData[i].rewards[k] + dqnOption.gamma * qValue - epsData[i].values[k];
				gaeValue = dqnOption.gamma * dqnOption.ppoLambda * gaeValue + delta;

				qValue = epsData[i].values[k];

				epsData[i].returns[k] = gaeValue + qValue;
				epsData[i].gaes[k] = gaeValue;

				//non gae
//				qValue = qValue * dqnOption.gamma + epsData[i].rewards[k];
//				epsData[i].returns[k] = qValue;
			}
			torch::Tensor gaeTensor = torch::from_blob(epsData[i].gaes.data(), {seqLen});
			torch::Tensor returnTensor = torch::from_blob(epsData[i].returns.data(), {seqLen, 1});
			gaeVec.push_back(gaeTensor);
			returnVec.push_back(returnTensor);

			torch::Tensor actionTensor = torch::from_blob(epsData[i].actions.data(), {seqLen, 1}, longOpt);
			actionVec.push_back(actionTensor);

			torch::Tensor oldPiTensor =  torch::from_blob(epsData[i].pis.data(), {seqLen});
			oldPiVec.push_back(oldPiTensor);

			std::vector<long> seqShape{seqLen};
			seqShape.insert(seqShape.end(), dqnOption.inputShape.begin(), dqnOption.inputShape.end());
			torch::Tensor stateTensor = torch::from_blob(epsData[i].states.data(), seqShape);
			inputStateVec.push_back(stateTensor);

			for (int gruIndex = 0; gruIndex < dqnOption.hidenLayerNums.size(); gruIndex ++) {
				hiddenStateVec[gruIndex].push_back(epsData[i].startState[gruIndex]);
			}
//			LOG4CXX_INFO(logger, "actionTensor " << actionTensor);
		}
//		torch::Tensor seqLensTensor = torch::from_blob(seqLens.data(), {seqLens.size()}, longOpt); //TODO: necessary?


//////////////////////////////////////////////////// Backward ////////////////////////////////////////////////////////
		for (int epochIndex = 0; epochIndex < dqnOption.epochNum; epochIndex ++) {
			auto indiceTensor = torch::randperm(dqnOption.trajStepNum, longOpt);
			long* indicePtr = indiceTensor.data_ptr<long>();

			for (int roundIndex = 0; roundIndex < roundNum; roundIndex ++) {
				updateNum ++;

////////////////////////////////////////////////// Round Input ///////////////////////////////////////////////////////
//				std::vector<long> seqLensPiece;
				std::vector<torch::Tensor> returnPieceVec;
				std::vector<torch::Tensor> actionPieceVec;
				std::vector<torch::Tensor> gaePieceVec;
				std::vector<torch::Tensor> oldPiPieceVec;
				std::vector<torch::Tensor> inputStatePieceVec;
				std::vector<std::vector<torch::Tensor>> hiddenStatePieceVec(dqnOption.gruCellNum);

				long minSeqLen = dqnOption.maxStep;
				for (int batchIndex = 0; batchIndex < dqnOption.batchSize; batchIndex ++) {
					minSeqLen = std::min(minSeqLen, seqLens[indicePtr[roundIndex * dqnOption.batchSize + batchIndex]]);
				}
//				LOG4CXX_INFO(logger, "------> min seq len: " << minSeqLen);

				for (int batchIndex = 0; batchIndex < dqnOption.batchSize; batchIndex ++) {
					int index = indicePtr[roundIndex * dqnOption.batchSize + batchIndex];

					returnPieceVec.push_back(returnVec[index].narrow(0, 0, minSeqLen));
					actionPieceVec.push_back(actionVec[index].narrow(0, 0, minSeqLen));
					gaePieceVec.push_back(gaeVec[index].narrow(0, 0, minSeqLen));
					oldPiPieceVec.push_back(oldPiVec[index].narrow(0, 0, minSeqLen));
					inputStatePieceVec.push_back(inputStateVec[index].narrow(0, 0, minSeqLen));

					for (int gruIndex = 0; gruIndex < dqnOption.gruCellNum; gruIndex ++) {
						hiddenStatePieceVec[gruIndex].push_back(hiddenStateVec[gruIndex][index]);
					}
				}

				torch::Tensor returnTensor = torch::cat(returnPieceVec, 0).view({minSeqLen * dqnOption.batchSize, 1}).to(dqnOption.deviceType);
				torch::Tensor gaeTensor = torch::cat(gaePieceVec, 0).view({minSeqLen * dqnOption.batchSize, 1}).to(dqnOption.deviceType);
				torch::Tensor oldPiTensor = torch::cat(oldPiPieceVec, 0).view({minSeqLen * dqnOption.batchSize, 1}).to(dqnOption.deviceType);
				torch::Tensor actionTensor = torch::cat(actionPieceVec, 0).view({minSeqLen * dqnOption.batchSize, 1}).to(dqnOption.deviceType);
				std::vector<long> seqShape{dqnOption.batchSize * minSeqLen};
				seqShape.insert(seqShape.end(), dqnOption.inputShape.begin(), dqnOption.inputShape.end());
				torch::Tensor stateTensor = torch::cat(inputStatePieceVec, 0).view(seqShape);
				stateTensor = stateTensor.div(dqnOption.inputScale).to(dqnOption.deviceType);
				std::vector<torch::Tensor> hiddenState;
				for (int gruIndex = 0; gruIndex < dqnOption.hidenLayerNums.size(); gruIndex ++) {
					hiddenState.push_back(torch::stack(hiddenStatePieceVec[gruIndex], 1));
				}
				LOG4CXX_DEBUG(logger, "returnTensor " << returnTensor.sizes());
				LOG4CXX_DEBUG(logger, "gaeTensor " << gaeTensor.sizes());
				LOG4CXX_DEBUG(logger, "oldPiTensor " << oldPiTensor.sizes());
				LOG4CXX_DEBUG(logger, "actionTensor " << actionTensor.sizes());

/////////////////////////////////////////////////// Round Output ///////////////////////////////////////////////////
				auto output = bModel.forwardNext(stateTensor, dqnOption.batchSize, minSeqLen, hiddenState, dqnOption.deviceType);
				torch::Tensor valueOutput = output[1];
				torch::Tensor actionOutput = output[0];

				torch::Tensor valueLossTensor = torch::nn::functional::mse_loss(valueOutput, returnTensor);

				torch::Tensor actionPiTensor = torch::softmax(actionOutput, -1);
				torch::Tensor actionPi = actionPiTensor.gather(-1, actionTensor);
				torch::Tensor ratio = actionPi / oldPiTensor;
//				LOG4CXX_INFO(logger, "ratio is " << ratio);
				float kl = ratio.mean().to(torch::kCPU).item<float>();

				auto advTensor = gaeTensor;
				auto sur0 = ratio * advTensor.detach();
				auto sur1 = torch::clamp(ratio, 1 - dqnOption.ppoEpsilon, 1 + dqnOption.ppoEpsilon) * advTensor.detach();
				LOG4CXX_DEBUG(logger, "sur0 = " << sur0.sizes());
				LOG4CXX_DEBUG(logger, "sur1 = " << sur1.sizes());
				torch::Tensor actLossTensor = torch::min(sur0, sur1).mean() * (-1);
				LOG4CXX_DEBUG(logger, "actLossTensor = " << actLossTensor);

				//non gae
//				torch::Tensor advTensor = (returnTensor - valueOutput);
//				torch::Tensor actionLogTensor = torch::log_softmax(actionOutput, -1); //{sum(seqLen), actionNum}
//				torch::Tensor actPiTensor = actionLogTensor.gather(-1, actionTensor);
//				torch::Tensor actLossTensor = - (actPiTensor * advTensor.detach()).mean();
//				LOG4CXX_INFO(logger, "advTensor " << advTensor.sizes());
//				LOG4CXX_INFO(logger, "actPiTensor " << actPiTensor.sizes());
//				LOG4CXX_INFO(logger, "actionPiTensor " << actionPiTensor.sizes());
//				LOG4CXX_DEBUG(logger, "actionPi " << actionPi.sizes());
//				LOG4CXX_INFO(logger, "ratio " << ratio.sizes());

				torch::Tensor actionLogDistTensor = torch::log_softmax(actionOutput, -1);
				LOG4CXX_DEBUG(logger, "actionLogTensor " << actionLogDistTensor.sizes());
				torch::Tensor entropyTensor = (-1) * (actionLogDistTensor * actionPiTensor).sum(-1).mean();
				LOG4CXX_DEBUG(logger, "entropy = " << entropyTensor);

				torch::Tensor lossTensor = actLossTensor + dqnOption.valueCoef * valueLossTensor - dqnOption.entropyCoef * entropyTensor;

				optimizer.zero_grad();
				lossTensor.backward();
				torch::nn::utils::clip_grad_norm_(bModel.parameters(), dqnOption.maxGradNormClip);
				optimizer.step();

				if ((updateNum % dqnOption.logInterval) == 0) {
					//print and log
					float lossV = lossTensor.item<float>();
					float vLossV = valueLossTensor.item<float>();
					float aLossV = actLossTensor.item<float>();
					float entropyV = entropyTensor.item<float>();
					float valueV = valueOutput.mean().item<float>();
					LOG4CXX_DEBUG(logger, "loss" << updateNum << "-" << epochIndex << "-" << roundIndex << ": " << lossV
							<< ", " << vLossV << ", " << aLossV << ", " << entropyV << ", " << kl);

					tLogger.add_scalar("loss/loss", step, lossV);
					tLogger.add_scalar("loss/vLoss", step, vLossV);
					tLogger.add_scalar("loss/aLoss", step, aLossV);
					tLogger.add_scalar("loss/entropy", step, entropyV);
					tLogger.add_scalar("loss/v", step, valueV);
					tLogger.add_scalar("loss/kl", step, kl);
					tLogger.add_scalar("loss/adv", step, advTensor.mean().item<float>());
				}
			}

		}


////////////////////////////////////////////////////// TEST /////////////////////////////////////////////////////////
		if ((updateNum % dqnOption.testGapEp) == 0) {
		if (dqnOption.toTest) {
			test();
		}
		}
	}

	save();
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PPOGRUTruncSlimGae<NetType, EnvType, PolicyType, OptimizerType>::save() {
	if (!dqnOption.saveModel) {
		return;
	}

	AlgUtils::SaveModel(bModel, optimizer, dqnOption.savePathPrefix, logger);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PPOGRUTruncSlimGae<NetType, EnvType, PolicyType, OptimizerType>::saveByReward(float reward) {
	if (!dqnOption.saveModel) {
		return;
	}

	std::string path = dqnOption.savePathPrefix + "_" + std::to_string(reward);
	AlgUtils::SaveModel(bModel, optimizer, path, logger);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PPOGRUTruncSlimGae<NetType, EnvType, PolicyType, OptimizerType>::load() {
	if (!dqnOption.loadModel) {
		return;
	}

	AlgUtils::LoadModel(bModel, optimizer, dqnOption.loadOptimizer, dqnOption.loadPathPrefix, logger);
}


#endif /* INC_ALG_RNN_PPOGRUTRUNCSLIMGAE_HPP_ */
