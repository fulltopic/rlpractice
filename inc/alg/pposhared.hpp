/*
 * ppo.hpp
 *
 *  Created on: Jun 23, 2021
 *      Author: zf
 */

#ifndef INC_ALG_PPOSHARED_HPP_
#define INC_ALG_PPOSHARED_HPP_


#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <vector>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/stats.h"
#include "gymtest/utils/lossstats.h"
#include "dqnoption.h"

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
class PPOShared {
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

	uint32_t updateNum = 0;
	const int updateTargetGap; //TODO


	std::vector<int64_t> batchInputShape;
	std::vector<int64_t> trajInputShape;
	int offset = 0;

//	const uint32_t trajStep;
//	const uint32_t epochNum;
//	uint32_t trajIndex = 0;
	const int actionNum;

	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("pposhared");

public:
	PPOShared(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, DqnOption option, int actNum);
	~PPOShared() = default;
	PPOShared(const PPOShared& ) = delete;

	void train(const int updateNum);
	void test(const int batchSize, const int epochNum);


	void load();
	void save();
};

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
PPOShared<NetType, EnvType, PolicyType, OptimizerType>::PPOShared(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy,
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
	testStater(iOption.statPathPrefix + "_test.txt", iOption.statCap),
	lossStater(iOption.statPathPrefix + "_loss.txt"),
	updateTargetGap(iOption.targetUpdate),
	actionNum(actNum){
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
}


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PPOShared<NetType, EnvType, PolicyType, OptimizerType>::train(const int updateNum) {
	LOG4CXX_INFO(logger, "training ");
	load();

	int updateIndex = 0;

	std::vector<float> statRewards(dqnOption.envNum, 0);
	std::vector<float> statLens(dqnOption.envNum, 0);

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

//		bModel.eval();

		//collect samples trajStepNum * envNum
		for (int trajIndex = 0; trajIndex < dqnOption.trajStepNum; trajIndex ++) {
			torch::Tensor stateTensor = torch::from_blob(stateVec.data(), inputShape).div(dqnOption.inputScale).to(deviceType);
//			LOG4CXX_INFO(logger, "stateTensor in step: " << stateTensor.max() << stateTensor.mean());
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

//			LOG4CXX_INFO(logger, "rewardVec" << step << ": " << rewardVec);
			std::vector<float> doneMaskVec(doneVec.size(), 1);
			for (int i = 0; i < doneVec.size(); i ++) {
//				LOG4CXX_INFO(logger, "dones: " << i << ": " << doneVec[i]);
				if (doneVec[i]) {
					doneMaskVec[i] = 0;

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
		LOG4CXX_DEBUG(logger, "Collect " << dqnOption.trajStepNum << " step samples ");

		//Calculate GAE return
		torch::Tensor lastStateTensor = torch::from_blob(stateVec.data(), inputShape).div(dqnOption.inputScale).to(deviceType);
		auto lastRc = bModel.forward(lastStateTensor);
		torch::Tensor lastValueTensor = lastRc[1];
		LOG4CXX_DEBUG(logger, "Get last value " << lastValueTensor.sizes());

		//TODO: Should squeeze the last 1?
		//TODO: Put all tensors in CPU to save storage for inputState tensors?
		at::IntArrayRef batchValueShape{dqnOption.trajStepNum, dqnOption.envNum, 1};

//		LOG4CXX_INFO(logger, "dones: " << donesVec.size() << ", " << donesVec[0].size());
//		LOG4CXX_INFO(logger, "dones numel: " << EnvUtils::FlattenVector(donesVec).size());
		auto doneData = EnvUtils::FlattenVector(donesVec);
		auto rewardData = EnvUtils::FlattenVector(rewardsVec);
//		LOG4CXX_INFO(logger, "done raw data ");
//		for (int i = 0; i < dqnOption.trajStepNum * dqnOption.envNum; i ++) {
//			LOG4CXX_INFO(logger, "doneData[" << i << "] = " << doneData[i]);
//		}
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
//			LOG4CXX_INFO(logger, "gae before: " << gaeReturn);
			torch::Tensor delta = rewardTensor[i] + dqnOption.gamma * nextValueTensor * doneTensor[i] - valueTensor[i];
			gaeReturn = delta + dqnOption.ppoLambda * dqnOption.gamma * gaeReturn * doneTensor[i];
//			LOG4CXX_INFO(logger, "reward: " << rewardTensor[i]);
//			LOG4CXX_INFO(logger, "done: " << doneTensor[i]);
//			LOG4CXX_INFO(logger, "nextValueTensor: "<< nextValueTensor);
//			LOG4CXX_INFO(logger, "valueTensor: " << valueTensor[i]);
//			LOG4CXX_INFO(logger, "delta: " << delta);
//			LOG4CXX_INFO(logger, "gaeReturn after: " << gaeReturn);
			plainReturn = rewardTensor[i] + dqnOption.gamma * plainReturn * doneTensor[i];

			gaeReturns[i].copy_(gaeReturn);
			returns[i].copy_(plainReturn);
			nextValueTensor = valueTensor[i];
		}

		//Put all tensors into GPU
		//normalize both gae and plain return
		gaeReturns = ((gaeReturns - gaeReturns.mean()) / (gaeReturns.std() + 1e-7)).to(deviceType).detach();
		LOG4CXX_DEBUG(logger, "Calculated GAE " << gaeReturns.sizes());
		//No normalization on return
		returns = returns.to(deviceType).detach();
//		returns = ((returns - returns.mean()) / (returns.std() + 1e-7)).to(deviceType).detach();
//		LOG4CXX_DEBUG(logger, "Calculated plain return " << returns.sizes());

		//Calculate old log Pi
		torch::Tensor oldDistTensor = torch::stack(pisVec, 0).view({dqnOption.trajStepNum, dqnOption.envNum, actionNum});
		oldDistTensor = torch::log_softmax(oldDistTensor, -1).to(deviceType);
		LOG4CXX_DEBUG(logger, "oldDistTensor: " << oldDistTensor.sizes());

		auto actionData = EnvUtils::FlattenVector(actionsVec);
		torch::Tensor oldActionTensor = torch::from_blob(actionData.data(), {dqnOption.trajStepNum, dqnOption.envNum, 1}, longOpt).to(deviceType);
		LOG4CXX_DEBUG(logger, "oldActionTensor: " << oldActionTensor.sizes());
		torch::Tensor oldPiTensor = oldDistTensor.gather(-1, oldActionTensor).detach();
		LOG4CXX_DEBUG(logger, "oldPiTensor: " << oldPiTensor.sizes());

		//Update
//		bModel.train();
		const int roundNum = dqnOption.trajStepNum / dqnOption.batchSize;

		auto stateData = EnvUtils::FlattenVector(statesVec);
		auto stateTensor = torch::from_blob(stateData.data(), trajInputShape).div(dqnOption.inputScale).to(deviceType);
		LOG4CXX_DEBUG(logger, "stateTensor: " << stateTensor.sizes());
		auto stateTensors = torch::chunk(stateTensor, roundNum, 0);
		auto gaeTensors = torch::chunk(gaeReturns, roundNum, 0);
		for (int i = 0; i < roundNum; i ++) {
			gaeTensors[i] = gaeTensors[i].view({dqnOption.batchSize * dqnOption.envNum, 1});
		}
		auto oldPiTensors = torch::chunk(oldPiTensor, roundNum, 0); //oldPiTensor = {trajStep, envNum, 1}
		for (int i = 0; i < roundNum; i ++) {
			oldPiTensors[i] = oldPiTensors[i].view({dqnOption.batchSize * dqnOption.envNum, 1});
		}
		auto actionTensors = torch::chunk(oldActionTensor, roundNum, 0);
		for (int i = 0; i < roundNum; i ++) {
			actionTensors[i] = actionTensors[i].view({dqnOption.batchSize * dqnOption.envNum, 1});
		}
		auto returnTensors = torch::chunk(returns, roundNum, 0);
		for (int i = 0; i < roundNum; i ++) {
			returnTensors[i] = returnTensors[i].view({dqnOption.batchSize * dqnOption.envNum, 1});
		}
		valueTensor = valueTensor.to(deviceType).detach();
		auto valueTensors = torch::chunk(valueTensor, roundNum, 0);
		for (int i = 0; i < roundNum; i ++) {
			valueTensors[i] = valueTensors[i].view({dqnOption.batchSize * dqnOption.envNum, 1});
		}

		std::vector<bool> toUpdate(roundNum, true);
		for (int epochIndex = 0; epochIndex < dqnOption.epochNum; epochIndex ++) {
			for (int roundIndex = 0; roundIndex < roundNum; roundIndex ++) {
				if (!toUpdate[roundIndex]) {
					LOG4CXX_DEBUG(logger, "-----------------------------------> early stop " << roundIndex);
					continue;
				} else {
					LOG4CXX_DEBUG(logger, "-------------------------------------> update one batch " << roundIndex);
				}

				LOG4CXX_DEBUG(logger, "train input " << stateTensors[roundIndex].sizes());
				auto rc = bModel.forward(stateTensors[roundIndex]);
				torch::Tensor valueOutput = rc[1];
				torch::Tensor actionOutput = rc[0];
				LOG4CXX_DEBUG(logger, "value output " << valueOutput);
				LOG4CXX_DEBUG(logger, " actionOutput " << actionOutput.sizes());

				//value loss
				if (dqnOption.valueClip) {
					valueOutput = valueTensors[roundIndex] + torch::clamp(valueOutput - valueTensors[roundIndex], (-1) * dqnOption.maxValueDelta, dqnOption.maxValueDelta);
				}
				torch::Tensor valueLossTensor = torch::nn::functional::mse_loss(valueOutput, returnTensors[roundIndex]);

				//action loss
				//actionPi and oldPi are logPi
				torch::Tensor advTensor = gaeTensors[roundIndex];
				LOG4CXX_DEBUG(logger, "advTensor: " << advTensor);
				torch::Tensor actionLogDistTensor = torch::log_softmax(actionOutput, -1);
				LOG4CXX_DEBUG(logger, "actionDistTensor: " << actionLogDistTensor);
				torch::Tensor actionPi = actionLogDistTensor.gather(-1, actionTensors[roundIndex]);
				LOG4CXX_DEBUG(logger, "actionPi: " << actionPi);
				auto ratio = torch::exp(actionPi - oldPiTensors[roundIndex]);
				LOG4CXX_DEBUG(logger, "ratio = " << ratio);


				auto piDelta = actionPi - oldPiTensors[roundIndex];
				auto kl = ((torch::exp(piDelta) - 1) - piDelta).mean().abs().to(torch::kCPU).item<float>();
				if (kl > dqnOption.maxKl) {
					LOG4CXX_INFO(logger, "Early stop: " << kl);
					toUpdate[roundIndex] = false;
					continue;
				}


				auto sur0 = ratio * advTensor;
				LOG4CXX_DEBUG(logger, "sur0 = " << sur0);
				auto sur1 = torch::clamp(ratio, 1 - dqnOption.ppoEpsilon, 1 + dqnOption.ppoEpsilon) * advTensor;
				LOG4CXX_DEBUG(logger, "sur1 = " << sur1);
				torch::Tensor actLossTensor = torch::min(sur0, sur1).mean() * (-1);
				LOG4CXX_DEBUG(logger, "actLossTensor = " << actLossTensor);

				//entropy loss
				torch::Tensor actionPiTensor = torch::softmax(actionOutput, -1);
				LOG4CXX_DEBUG(logger, "actionLogTensor " << actionPiTensor);
				torch::Tensor entropyTensor = (-1) * (actionLogDistTensor * actionPiTensor).sum(-1).mean();
				LOG4CXX_DEBUG(logger, "entropy = " << entropyTensor);

				torch::Tensor lossTensor = actLossTensor + dqnOption.valueCoef * valueLossTensor - dqnOption.entropyCoef * entropyTensor;

				//print and log
				auto lossV = lossTensor.item<float>();
				auto vLossV = valueLossTensor.item<float>();
				auto aLossV = actLossTensor.item<float>();
				auto entropyV = entropyTensor.item<float>();
				LOG4CXX_INFO(logger, "loss" << updateIndex << "-" << epochIndex << "-" << roundIndex << ": " << lossV
						<< ", " << vLossV << ", " << aLossV << ", " << entropyV << ", " << kl);
//				LOG4CXX_INFO(logger, "ratio" << )

				auto curState = stater.getCurState();
				lossStater.update({updateIndex, epochIndex, roundIndex, lossV, vLossV, aLossV, entropyV, kl,
					curState[0], curState[1]});
//				torch::Tensor lossTensor = actLossTensor;

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
void PPOShared<NetType, EnvType, PolicyType, OptimizerType>::test(const int batchSize, const int epochNum) {
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
void PPOShared<NetType, EnvType, PolicyType, OptimizerType>::save() {
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
void PPOShared<NetType, EnvType, PolicyType, OptimizerType>::load() {
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

#endif /* INC_ALG_PPOSHARED_HPP_ */
