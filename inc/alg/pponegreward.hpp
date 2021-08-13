/*
 * pponegreward.hpp
 *
 *  Created on: Aug 6, 2021
 *      Author: zf
 */

#ifndef INC_ALG_PPONEGREWARD_HPP_
#define INC_ALG_PPONEGREWARD_HPP_

#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/stats.h"
#include "gymtest/utils/lossstats.h"
#include "dqnoption.h"

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
class PPOPenalty {
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
	Stats stepStater;
	Stats testStater;
	Stats idleStepStater;
	LossStats lossStater;

	uint32_t updateNum = 0;
	const int updateTargetGap; //TODO


	std::vector<int64_t> batchInputShape;
	std::vector<int64_t> trajInputShape;
	int offset = 0;

//	const uint32_t trajStep;
//	const uint32_t epochNum;
//	uint32_t trajIndex = 0;
	const int actionNum;

	std::vector<int64_t> indice;

	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("pposhared");

public:
	PPOPenalty(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, DqnOption option, int actNum);
	~PPOPenalty() = default;
	PPOPenalty(const PPOPenalty& ) = delete;

	void train(const int updateNum);
	void test(const int batchSize, const int epochNum);


	void load();
	void save();
};

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
PPOPenalty<NetType, EnvType, PolicyType, OptimizerType>::PPOPenalty(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy,
		OptimizerType& iOptimizer,
		const DqnOption iOption,
		int actNum):
	bModel(behaviorModel),
//	tModel(trainModel),
	env(iEnv),
	testEnv(tEnv),
	policy(iPolicy),
	optimizer(iOptimizer),
	dqnOption(iOption),
	deviceType(iOption.deviceType),
	inputShape(iOption.inputShape),
	stater(iOption.statPathPrefix + "_stat.txt", iOption.statCap),
	stepStater(iOption.statPathPrefix + "_step.txt", iOption.statCap),
	testStater(iOption.statPathPrefix + "_test.txt", iOption.statCap),
	idleStepStater(iOption.statPathPrefix + "_idle.txt", iOption.statCap),
	lossStater(iOption.statPathPrefix + "_loss.txt"),
	updateTargetGap(iOption.targetUpdate),
	actionNum(actNum),
	indice(iOption.trajStepNum * iOption.envNum, 0){
//	if (dqnOption.isAtari) {
		batchInputShape.push_back(dqnOption.envNum * dqnOption.batchSize);
		trajInputShape.push_back(dqnOption.trajStepNum * dqnOption.envNum);
		for (int i = 1; i < inputShape.size(); i ++) {
			batchInputShape.push_back(inputShape[i]);
			trajInputShape.push_back(inputShape[i]);
		}
//	}
//	else {
//		batchInputShape.push_back(dqn);
//		for (int i = 0; i < inputShape.size(); i ++) {
//			batchInputShape.push_back(inputShape[i]);
//		}
//	}

	offset = dqnOption.batchSize;
	for (int i = 0; i < inputShape.size(); i ++) {
		offset *= inputShape[i];
	}

	for (int i = 0; i < iOption.trajStepNum * iOption.envNum; i ++) {
		indice[i] = i;
	}
//	LOG4CXX_INFO(logger, "indice after initiation \n " << indice);
	std::srand(unsigned (std::time(0)));
}


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PPOPenalty<NetType, EnvType, PolicyType, OptimizerType>::train(const int updateNum) {
	LOG4CXX_INFO(logger, "training ");
	load();

	int updateIndex = 0;

	std::vector<float> statRewards(dqnOption.envNum, 0);
	std::vector<float> statLens(dqnOption.envNum, 0);
	std::vector<int> liveCounts(dqnOption.envNum, 0);
	std::vector<float> sumRewards(dqnOption.envNum, 0);
	std::vector<float> sumLens(dqnOption.envNum, 0);
	std::vector<float> clipRewards(dqnOption.envNum, 0);
	std::vector<float> clipSumRewards(dqnOption.envNum, 0);
	std::vector<float> idleStep(dqnOption.envNum, 0);

	std::vector<float> stateVec = env.reset();
	while (updateIndex < updateNum) {
		LOG4CXX_INFO(logger, "---------------------------------------> update  " << updateIndex);

		updateIndex ++;

		std::vector<std::vector<float>> statesVec;
		std::vector<std::vector<float>> rewardsVec;
		std::vector<std::vector<float>> donesVec;
		std::vector<std::vector<long>> actionsVec;
		std::vector<torch::Tensor> valuesVec;
		std::vector<torch::Tensor> pisVec;

		bModel.eval();

		//collect samples trajStepNum * envNum
		for (int trajIndex = 0; trajIndex < dqnOption.trajStepNum; trajIndex ++) {
			torch::Tensor stateTensor = torch::from_blob(stateVec.data(), inputShape).div(dqnOption.inputScale).to(deviceType);
			std::vector<torch::Tensor> rc = bModel.forward(stateTensor);
			valuesVec.push_back(rc[1]);
			pisVec.push_back(rc[0]);

			auto actionProbs =  torch::softmax(rc[0], -1);
			std::vector<int64_t> actions = policy.getActions(actionProbs);

			auto stepResult = env.step(actions, false);
			auto nextStateVec = std::get<0>(stepResult);
			auto rewardVec = std::get<1>(stepResult);
			auto doneVec = std::get<2>(stepResult);

			Stats::UpdateReward(statRewards, rewardVec);
			Stats::UpdateLen(statLens);
			if (dqnOption.clipRewardStat) {
				Stats::UpdateReward(clipRewards, rewardVec, true, dqnOption.rewardMin, dqnOption.rewardMax);
			}

			std::vector<float> doneMaskVec(doneVec.size(), 1);
			for (int i = 0; i < doneVec.size(); i ++) {
				if (doneVec[i]) {
					doneMaskVec[i] = 0;

					sumRewards[i] += statRewards[i];
					sumLens[i] += statLens[i];
					stater.update(statLens[i], statRewards[i]);
					if (dqnOption.clipRewardStat) {
						clipSumRewards[i] += clipRewards[i];
						stepStater.update(statLens[i], clipRewards[i]);
						clipRewards[i] = 0;
					}
					statLens[i] = 0;
					statRewards[i] = 0;
					LOG4CXX_INFO(logger, "c" << i << stater << " --- " << stepStater);

					//End of living
					if (dqnOption.toPunish) {
						idleStepStater.update(idleStep[i], sumRewards[i]);
						idleStep[i] = 0;
					}

					if (dqnOption.multiLifes) {
						liveCounts[i] ++;
						if (liveCounts[i] >= dqnOption.donePerEp) {
//							LOG4CXX_INFO(logger, "Wrapper episode " << i << " ----------------------------> " << sumRewards[i]);

							liveCounts[i] = 0;
							sumRewards[i] = 0;
							sumLens[i] = 0;
							clipSumRewards[i] = 0;
						}
					}
				}
			}

			if (dqnOption.toPunish) {
				for (int i = 0; i < rewardVec.size(); i ++) {
					if (rewardVec[i] <= 0.1) {
						idleStep[i] ++;
						if (idleStep[i] >= dqnOption.penalStep) {
							rewardVec[i] = dqnOption.penalReward;
							idleStepStater.update(idleStep[i], sumRewards[i]);

							idleStep[i] = 0;
							LOG4CXX_INFO(logger, "=============================================>c" << i << " punished: " << dqnOption.penalReward << " after " << dqnOption.penalStep);
						}
					} else { //get reward
						idleStepStater.update(idleStep[i], sumRewards[i]);
						idleStep[i] = 0;
					}
				}
			}


			statesVec.push_back(stateVec);
			rewardsVec.push_back(rewardVec);
			donesVec.push_back(doneMaskVec);
			actionsVec.push_back(actions);

			stateVec = nextStateVec;
		}
		LOG4CXX_DEBUG(logger, "Collect " << dqnOption.trajStepNum << " step samples ");

		//Calculate GAE return
		torch::Tensor lastStateTensor = torch::from_blob(stateVec.data(), inputShape).div(dqnOption.inputScale).to(deviceType);
		auto lastRc = bModel.forward(lastStateTensor);
		torch::Tensor lastValueTensor = lastRc[1];
		LOG4CXX_DEBUG(logger, "Get last value " << lastValueTensor.sizes());

		//TODO: Should squeeze the last 1?
		//TODO: Put all tensors in CPU to save storage for inputState tensors?
		at::IntArrayRef batchValueShape{dqnOption.trajStepNum, dqnOption.envNum, 1};

		auto doneData = EnvUtils::FlattenVector(donesVec);
		auto rewardData = EnvUtils::FlattenVector(rewardsVec);

		//TODO: Check batchValueShape matches original layout
		torch::Tensor valueTensor = torch::stack(valuesVec, 0).view(batchValueShape).to(torch::kCPU); //toCPU necessary?
		torch::Tensor rewardTensor = torch::from_blob(rewardData.data(), batchValueShape).div(dqnOption.rewardScale).clamp(dqnOption.rewardMin, dqnOption.rewardMax);
		torch::Tensor doneTensor = torch::from_blob(doneData.data(), batchValueShape);
		LOG4CXX_DEBUG(logger, "dones " << "\n" << doneTensor);
		torch::Tensor gaeReturns = torch::zeros(batchValueShape);
		torch::Tensor returns = torch::zeros(batchValueShape);
		torch::Tensor nextValueTensor = lastValueTensor.to(torch::kCPU).detach();
		torch::Tensor gaeReturn = torch::zeros({dqnOption.envNum, 1});
		torch::Tensor plainReturn = torch::zeros({dqnOption.envNum, 1});
		plainReturn.copy_(nextValueTensor);
		LOG4CXX_DEBUG(logger, "nextValueTensor " << nextValueTensor.sizes());

		LOG4CXX_DEBUG(logger, "--------------------------------------> Calculate GAE");
		//tmp solution
		torch::Tensor returnDoneTensor = torch::zeros(batchValueShape);
		returnDoneTensor.copy_(doneTensor);
		for (int i = dqnOption.trajStepNum - 1; i >= 0; i --) {
			torch::Tensor delta = rewardTensor[i] + dqnOption.gamma * nextValueTensor * doneTensor[i] - valueTensor[i];
			gaeReturn = delta + dqnOption.ppoLambda * dqnOption.gamma * gaeReturn * doneTensor[i];
//			plainReturn = rewardTensor[i] + dqnOption.gamma * plainReturn * doneTensor[i];

			gaeReturns[i].copy_(gaeReturn);
//			returns[i].copy_(plainReturn);
			nextValueTensor = valueTensor[i];
		}
		returns = gaeReturns + valueTensor; //from baseline3. TODO: why?

		//Put all tensors into GPU
		gaeReturns = gaeReturns.to(deviceType).detach();
		LOG4CXX_DEBUG(logger, "Calculated GAE " << gaeReturns.sizes());
		returns = returns.to(deviceType).detach();

		//Calculate old log Pi
		torch::Tensor oldDistTensor = torch::stack(pisVec, 0).view({dqnOption.trajStepNum, dqnOption.envNum, actionNum});
		oldDistTensor = torch::softmax(oldDistTensor, -1).to(deviceType);
		LOG4CXX_DEBUG(logger, "oldDistTensor: " << oldDistTensor.sizes());

		auto actionData = EnvUtils::FlattenVector(actionsVec);
		torch::Tensor oldActionTensor = torch::from_blob(actionData.data(), {dqnOption.trajStepNum, dqnOption.envNum, 1}, longOpt).to(deviceType);
		LOG4CXX_DEBUG(logger, "oldActionTensor: " << oldActionTensor.sizes());
		torch::Tensor oldPiTensor = oldDistTensor.gather(-1, oldActionTensor).detach();
		LOG4CXX_DEBUG(logger, "oldPiTensor: " << oldPiTensor.sizes());

		//Update
		bModel.train();
		//TODO: update batchSize config
		const int roundNum = dqnOption.trajStepNum / dqnOption.batchSize;

		auto stateData = EnvUtils::FlattenVector(statesVec);
		auto stateTensor = torch::from_blob(stateData.data(), trajInputShape).div(dqnOption.inputScale).to(deviceType);
		LOG4CXX_DEBUG(logger, "stateTensor: " << stateTensor.sizes());

//		auto gaeTensors = torch::split(gaeReturns.view({roundNum * dqnOption.batchSize * dqnOption.envNum, 1}), dqnOption.batchSize * dqnOption.envNum);
//		auto oldPiTensors = torch::split(oldPiTensor.view({roundNum * dqnOption.batchSize * dqnOption.envNum, 1}), dqnOption.batchSize * dqnOption.envNum);
//		auto actionTensors = torch::split(oldActionTensor.view({roundNum * dqnOption.batchSize * dqnOption.envNum, 1}), dqnOption.batchSize * dqnOption.envNum);
//		auto returnTensors = torch::split(returns.view({roundNum * dqnOption.batchSize * dqnOption.envNum, 1}), dqnOption.batchSize * dqnOption.envNum);
//		auto valueTensors = torch::split(valueTensor.view({roundNum * dqnOption.batchSize * dqnOption.envNum, 1}), dqnOption.batchSize * dqnOption.envNum);
		gaeReturns = gaeReturns.view({dqnOption.trajStepNum * dqnOption.envNum, 1});
		oldPiTensor = oldPiTensor.view({dqnOption.trajStepNum * dqnOption.envNum, 1});
		oldActionTensor = oldActionTensor.view({dqnOption.trajStepNum * dqnOption.envNum, 1});
		returns = returns.view({dqnOption.trajStepNum * dqnOption.envNum, 1});
		valueTensor = gaeReturns.view({dqnOption.trajStepNum * dqnOption.envNum, 1});


//		LOG4CXX_INFO(logger, "shuffled indice " << indice);
//		LOG4CXX_INFO(logger, "indiceTensor \n" << indiceTensor);
		auto pieceLen = dqnOption.batchSize * dqnOption.envNum;

//		auto indiceTensor = torch::randperm(dqnOption.trajStepNum * dqnOption.envNum, longOpt).view({-1, dqnOption.batchSize * dqnOption.envNum}).to(deviceType);
		for (int epochIndex = 0; epochIndex < dqnOption.epochNum; epochIndex ++) {
//			LOG4CXX_INFO(logger, "epoch " << epochIndex << " state1 = \n" << stateTensor[1]);
//			LOG4CXX_INFO(logger, "stateVec0 = " << statesVec[0]);
			//shuffle index
//			std::random_shuffle(indice.begin(), indice.end());
//			torch::Tensor indiceTensor = torch::from_blob(indice.data(), {indice.size()}, longOpt).to(deviceType);
			auto indiceTensor = torch::randperm(dqnOption.trajStepNum * dqnOption.envNum, longOpt).view({-1, dqnOption.batchSize * dqnOption.envNum}).to(deviceType);

			for (int roundIndex = 0; roundIndex < roundNum; roundIndex ++) {
				//fetch data
//				torch::Tensor indexPiece = indiceTensor.narrow(0, pieceLen * roundIndex, pieceLen);
				auto indexPiece = indiceTensor[roundIndex];
//				LOG4CXX_INFO(logger, "index piece: " << indexPiece);
//				LOG4CXX_INFO(logger, "state size: " << stateTensor.sizes());
				torch::Tensor stateInput = stateTensor.index_select(0, indexPiece);
//				LOG4CXX_INFO(logger, "state input sizes: " << stateInput.sizes());
				torch::Tensor returnPiece = returns.index_select(0, indexPiece);
				torch::Tensor gaePiece = gaeReturns.index_select(0, indexPiece);
				torch::Tensor oldPiPiece = oldPiTensor.index_select(0, indexPiece);
				torch::Tensor oldActionPiece = oldActionTensor.index_select(0, indexPiece);
				torch::Tensor valuePiece = valueTensor.index_select(0, indexPiece);

				auto rc = bModel.forward(stateInput);
				torch::Tensor valueOutput = rc[1];
				torch::Tensor actionOutput = rc[0];


				torch::Tensor valueLossTensor = torch::nn::functional::mse_loss(valueOutput, returnPiece);

				//action loss
				//actionPi and oldPi are logPi
				torch::Tensor advTensor = gaePiece;
				if (dqnOption.normReward) {
					advTensor = (advTensor - advTensor.mean()) / (advTensor.std() + 1e-7);
				}
				torch::Tensor actionPiTensor = torch::softmax(actionOutput, -1);
				torch::Tensor actionPi = actionPiTensor.gather(-1, oldActionPiece);
				auto ratio = actionPi / oldPiPiece;
//				LOG4CXX_INFO(logger, "ratio is " << ratio);
				auto kl = ratio.mean().to(torch::kCPU).item<float>();


				auto sur0 = ratio * advTensor.detach();
				auto sur1 = torch::clamp(ratio, 1 - dqnOption.ppoEpsilon, 1 + dqnOption.ppoEpsilon) * advTensor.detach();
				LOG4CXX_DEBUG(logger, "sur1 = " << sur1);
				torch::Tensor actLossTensor = torch::min(sur0, sur1).mean() * (-1);
				LOG4CXX_DEBUG(logger, "actLossTensor = " << actLossTensor);

				//entropy loss
				torch::Tensor actionLogDistTensor = torch::log_softmax(actionOutput, -1);
				LOG4CXX_DEBUG(logger, "actionLogTensor " << actionLogDistTensor);
				torch::Tensor entropyTensor = (-1) * (actionLogDistTensor * actionPiTensor).sum(-1).mean();
				LOG4CXX_DEBUG(logger, "entropy = " << entropyTensor);

				//overall loss
				torch::Tensor lossTensor = actLossTensor + dqnOption.valueCoef * valueLossTensor - dqnOption.entropyCoef * entropyTensor;
//				auto lossTensor = sur0.mean() * (-1);
//				auto lossTensor = actLossTensor - dqnOption.entropyCoef * entropyTensor;

				//print and log
				auto lossV = lossTensor.item<float>();
				auto vLossV = valueLossTensor.item<float>();
				auto aLossV = actLossTensor.item<float>();
				auto entropyV = entropyTensor.item<float>();
				LOG4CXX_INFO(logger, "loss" << updateIndex << "-" << epochIndex << "-" << roundIndex << ": " << lossV
						<< ", " << vLossV << ", " << aLossV << ", " << entropyV << ", " << kl);

				auto curState = stater.getCurState();
				lossStater.update({updateIndex, epochIndex, roundIndex, lossV, vLossV, aLossV, entropyV, kl,
					curState[0], curState[1]});

				optimizer.zero_grad();
				lossTensor.backward();
				torch::nn::utils::clip_grad_norm_(bModel.parameters(), dqnOption.maxGradNormClip);
				optimizer.step();
			}
		}
	}

	save();
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PPOPenalty<NetType, EnvType, PolicyType, OptimizerType>::test(const int batchSize, const int epochNum) {
	LOG4CXX_INFO(logger, "To test " << epochNum << " episodes");
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
void PPOPenalty<NetType, EnvType, PolicyType, OptimizerType>::save() {
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
void PPOPenalty<NetType, EnvType, PolicyType, OptimizerType>::load() {
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



#endif /* INC_ALG_PPONEGREWARD_HPP_ */