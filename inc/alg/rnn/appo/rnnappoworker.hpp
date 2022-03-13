/*
 * rnnappoworker.hpp
 *
 *  Created on: Mar 5, 2022
 *      Author: zf
 */

#ifndef INC_ALG_RNN_APPO_RNNAPPOWORKER_HPP_
#define INC_ALG_RNN_APPO_RNNAPPOWORKER_HPP_


#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <tensorboard_logger.h>


#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>

#include "alg/utils/dqnoption.h"
#include "gymtest/env/envutils.h"
#include "gymtest/utils/stats.h"
//#include "gymtest/utils/lossstats.h"
#include "alg/rnn/appo/rnnappodataq.h"

template<typename NetType, typename EnvType, typename PolicyType>
class RnnAPPOWorker {
private:
	NetType& bModel;
	EnvType& env;
	PolicyType& policy;
	AsyncRnnPPODataQ& q;

	const torch::Device deviceType;
	const at::IntArrayRef inputShape;

	const DqnOption dqnOption;
	torch::TensorOptions devOpt;
	torch::TensorOptions devLongOpt;

	uint32_t updateNum = 0;
	const int updateTargetGap; //TODO


	std::vector<int64_t> stepConvInputShape;
	uint32_t bulkSize = 1;

	std::vector<std::vector<float>> statesVec;
	std::vector<std::vector<float>> rewardsVec;
	std::vector<std::vector<long>> actionsVec;
	std::vector<std::vector<float>> valuesVec;
	std::vector<std::vector<float>> pisVec;
	std::vector<std::vector<torch::Tensor>> hStateVec;

	const int actionNum;

	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("testappornn");
	TensorBoardLogger tLogger;

	void processRecord(const int index, float lastValue, std::vector<torch::Tensor>& hStates);

public:
	RnnAPPOWorker(NetType& behaviorModel, EnvType& iEnv, PolicyType& iPolicy,
			DqnOption option, AsyncRnnPPODataQ& iq, int actNum);
	~RnnAPPOWorker() = default;
	RnnAPPOWorker(const RnnAPPOWorker& ) = delete;

	void train(const int updateNum);


	void load();
	void save();
};

template<typename NetType, typename EnvType, typename PolicyType>
RnnAPPOWorker<NetType, EnvType, PolicyType>::RnnAPPOWorker(NetType& behaviorModel, EnvType& iEnv, PolicyType& iPolicy,
		const DqnOption iOption,
		AsyncRnnPPODataQ& iq,
		int actNum):
	bModel(behaviorModel),
	env(iEnv),
	policy(iPolicy),
	q(iq),
	dqnOption(iOption),
	deviceType(iOption.deviceType),
	inputShape(iOption.inputShape),
	updateTargetGap(iOption.targetUpdate),
	tLogger(iOption.tensorboardLogPath.c_str()),
	actionNum(actNum)
{
	stepConvInputShape.push_back(dqnOption.envNum);
	stepConvInputShape.push_back(1);
	for(auto& inputDim: inputShape) {
		stepConvInputShape.push_back(inputDim);
		bulkSize *= inputDim;
	}

	devOpt = torch::TensorOptions().device(dqnOption.deviceType);
	devLongOpt = torch::TensorOptions().device(dqnOption.deviceType).dtype(torch::kLong);

	statesVec = std::vector<std::vector<float>>(dqnOption.envNum);
	rewardsVec = std::vector<std::vector<float>>(dqnOption.envNum);
	actionsVec = std::vector<std::vector<long>>(dqnOption.envNum);
	valuesVec = std::vector<std::vector<float>>(dqnOption.envNum);
	pisVec = std::vector<std::vector<float>>(dqnOption.envNum);
	hStateVec = std::vector<std::vector<torch::Tensor>>(dqnOption.envNum);
}

template<typename NetType, typename EnvType, typename PolicyType>
void RnnAPPOWorker<NetType, EnvType, PolicyType>::processRecord(const int index, float lastValue, std::vector<torch::Tensor>& hStates) {
	const int seqLen = actionsVec[index].size();
	std::vector<long> seqShape{seqLen};
	for (const auto& inputDim: dqnOption.inputShape) {
		seqShape.push_back(inputDim);
	}

	//2. generate tensors
	torch::Tensor stateTensor = torch::zeros(seqShape, devOpt);
	torch::Tensor actionTensor = torch::zeros({seqLen, 1}, devLongOpt);
	torch::Tensor gaeTensor = torch::zeros({seqLen, 1}, devOpt);
	torch::Tensor returnTensor = torch::zeros({seqLen, 1}, devOpt);
	torch::Tensor oldPiTensor = torch::zeros({seqLen, 1}, devOpt);
	//copy of tensor vector, suppose copy of tensor
	std::vector<torch::Tensor> hState = hStateVec[index]; //TODO: copy required?

	torch::Tensor cpuState = torch::from_blob(statesVec[index].data(), seqShape); //TODO: scale?
	torch::Tensor cpuAction = torch::from_blob(actionsVec[index].data(), {seqLen, 1}, longOpt);
	torch::Tensor cpuOldPi = torch::from_blob(pisVec[index].data(), {seqLen, 1});

	stateTensor.copy_(cpuState);
	actionTensor.copy_(cpuAction);
	oldPiTensor.copy_(cpuOldPi);

	//3. calculate gae
	float qValue = lastValue;
	float gaeValue = 0;
	for (int i = seqLen - 1; i >= 0; i --) {
		float delta = rewardsVec[index][i] + dqnOption.gamma * qValue - valuesVec[index][i];
		gaeValue = dqnOption.gamma * dqnOption.ppoLambda * gaeValue + delta;

		qValue = valuesVec[index][i];

		returnTensor[i][0] = gaeValue + qValue;
		gaeTensor[i][0] = gaeValue;
	}

	//4. push into q
	std::vector<torch::Tensor> rc{stateTensor, actionTensor, returnTensor, gaeTensor, oldPiTensor};
	rc.insert(rc.end(), hState.begin(), hState.end());
	q.push(std::move(rc));

	//5. reset vecs (e.g. statesVec)
	statesVec[index] = std::vector<float>();
	rewardsVec[index] = std::vector<float>();
	actionsVec[index] = std::vector<long>();
	valuesVec[index] = std::vector<float>();
	pisVec[index] = std::vector<float>();
	hStateVec[index] = bModel.getHState(index, hStates);
}

template<typename NetType, typename EnvType, typename PolicyType>
void RnnAPPOWorker<NetType, EnvType, PolicyType>::train(const int stepNum) {
	torch::NoGradGuard guard;

	LOG4CXX_INFO(logger, "training: is multi " << dqnOption.multiLifes << " lives " << dqnOption.donePerEp);
	load();

//	int updateIndex = 0;
	uint64_t stepCount = 0;
	int epCount = 0;
	int roundCount = 0;

	std::vector<float> statRewards(dqnOption.envNum, 0);
	std::vector<float> statLens(dqnOption.envNum, 0);
	std::vector<float> sumRewards(dqnOption.envNum, 0);
	std::vector<float> sumLens(dqnOption.envNum, 0);
	std::vector<int> liveCounts(dqnOption.envNum, 0);


	std::vector<torch::Tensor> stepStates = bModel.createHStates(dqnOption.envNum, dqnOption.deviceType);
	for (int index = 0; index < dqnOption.envNum; index ++) {
		hStateVec[index] = bModel.getHState(index, stepStates);
	}

	std::vector<float> stateVec = env.reset();
	while (stepCount < stepNum) {
		stepCount ++;
//		LOG4CXX_INFO(logger, "---------------------------------------> update  " << updateIndex);


		torch::Tensor stateTensor = torch::from_blob(stateVec.data(), stepConvInputShape).div(dqnOption.inputScale).to(deviceType);
		auto rc = bModel.forwardNext(stateTensor, stepStates);
		torch::Tensor rcValue = std::get<0>(rc)[1];
		torch::Tensor rcAction = std::get<0>(rc)[0];
		std::vector<torch::Tensor> nextStepStates = std::get<1>(rc);

		auto actionProbs = torch::softmax(rcAction, -1);
		std::vector<int64_t> actions = policy.getActions(actionProbs);

		auto stepResult = env.step(actions, false);
		auto nextStateVec = std::get<0>(stepResult);
		auto rewardVec = std::get<1>(stepResult);
		auto doneVec = std::get<2>(stepResult);

		for (int index = 0; index < dqnOption.envNum; index ++) {
			if (actionsVec[index].size() >= dqnOption.maxStep) {
				float lastValue = rcValue[index].item<float>();
				processRecord(index, lastValue, stepStates);
			}
		}

		for (int index = 0; index < dqnOption.envNum; index ++) {
			statesVec[index].insert(statesVec[index].end(), stateVec.begin() + bulkSize * index, stateVec.begin() + bulkSize * (index + 1));
			actionsVec[index].push_back(actions[index]);
			rewardsVec[index].push_back(std::min(std::max(dqnOption.rewardMin, rewardVec[index] / dqnOption.rewardScale), dqnOption.rewardMax));
			valuesVec[index].push_back(rcValue[index].item<float>());
			pisVec[index].push_back(actionProbs[index][actions[index]].item<float>());
		}

		Stats::UpdateReward(statRewards, rewardVec);
		Stats::UpdateLen(statLens);

		for (int index = 0; index < doneVec.size(); index ++) {
//					LOG4CXX_INFO(logger, "dones: " << i << ": " << doneVec[i]);
			if (doneVec[index]) {
				epCount ++;

				sumRewards[index] += statRewards[index];
				sumLens[index] += statLens[index];

				tLogger.add_scalar("train/len", epCount, statLens[index]);
				tLogger.add_scalar("train/reward", epCount, statRewards[index]);
				LOG4CXX_INFO(logger, "ep " << stepCount << ": " << statLens[index] << ", " << statRewards[index]);

				statLens[index] = 0;
				statRewards[index] = 0;

				if (dqnOption.multiLifes) {
					liveCounts[index] ++;
					if (liveCounts[index] >= dqnOption.donePerEp) {
						roundCount ++;
						LOG4CXX_INFO(logger, "Wrapper episode " << index << " ----------------------------> " << sumRewards[index]);
						tLogger.add_scalar("train/sumLen", roundCount, sumLens[index]);
						tLogger.add_scalar("train/sumReward", roundCount, sumRewards[index]);

						liveCounts[index] = 0;
						sumRewards[index] = 0;
						sumLens[index] = 0;
					}
				}

				//RNN
				//TODO: To reset for breakout?
				bModel.resetHState(index, nextStepStates);
				processRecord(index, 0, nextStepStates);
			}
		}

		stateVec = nextStateVec;
		stepStates = nextStepStates;


		if ((stepCount % dqnOption.logInterval) == 0) {
			tLogger.add_scalar("train/t_step", stepCount, (float)stepCount);
		}

	}

	save();
}

template<typename NetType, typename EnvType, typename PolicyType>
void RnnAPPOWorker<NetType, EnvType, PolicyType>::save() {
	if (!dqnOption.saveModel) {
		return;
	}

	std::string modelPath = dqnOption.savePathPrefix + "_model.pt";
	torch::serialize::OutputArchive outputArchive;
	bModel.save(outputArchive);
	outputArchive.save_to(modelPath);
	LOG4CXX_INFO(logger, "Save model into " << modelPath);
}

template<typename NetType, typename EnvType, typename PolicyType>
void RnnAPPOWorker<NetType, EnvType, PolicyType>::load() {
	if (!dqnOption.loadModel) {
		return;
	}

	std::string modelPath = dqnOption.loadPathPrefix + "_model.pt";
	torch::serialize::InputArchive inChive;
	inChive.load_from(modelPath);
	bModel.load(inChive);
	LOG4CXX_INFO(logger, "Load model from " << modelPath);

}


#endif /* INC_ALG_RNN_APPO_RNNAPPOWORKER_HPP_ */
