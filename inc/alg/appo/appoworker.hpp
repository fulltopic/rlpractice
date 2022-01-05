/*
 * appoworker.hpp
 *
 *  Created on: Dec 15, 2021
 *      Author: zf
 */

#ifndef INC_ALG_APPO_APPOWORKER_HPP_
#define INC_ALG_APPO_APPOWORKER_HPP_


#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <tensorboard_logger.h>


#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>

#include "../utils/dqnoption.h"
#include "gymtest/env/envutils.h"
#include "gymtest/utils/stats.h"
#include "gymtest/utils/lossstats.h"
#include "appodataq.h"

template<typename NetType, typename EnvType, typename PolicyType>
class APPOWorker {
private:
	NetType& bModel;
	EnvType& env;
	PolicyType& policy;
	AsyncPPODataQ& q;

	const torch::Device deviceType;
	const at::IntArrayRef inputShape;

	const DqnOption dqnOption;

	uint32_t updateNum = 0;
	const int updateTargetGap; //TODO


//	std::vector<int64_t> batchInputShape;
	std::vector<int64_t> trajInputShape;

	const int actionNum;

	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("pposhared");
	TensorBoardLogger tLogger;

public:
	APPOWorker(NetType& behaviorModel, EnvType& iEnv, PolicyType& iPolicy,
			DqnOption option, AsyncPPODataQ& iq, int actNum);
	~APPOWorker() = default;
	APPOWorker(const APPOWorker& ) = delete;

	void train(const int updateNum);
//	void test(const int batchSize, const int epochNum);


	void load();
	void save();
};

template<typename NetType, typename EnvType, typename PolicyType>
APPOWorker<NetType, EnvType, PolicyType>::APPOWorker(NetType& behaviorModel, EnvType& iEnv, PolicyType& iPolicy,
		const DqnOption iOption,
		AsyncPPODataQ& iq,
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

//	batchInputShape.push_back(dqnOption.envNum * dqnOption.batchSize);
	trajInputShape.push_back(dqnOption.trajStepNum * dqnOption.envNum);
	for (int i = 1; i < inputShape.size(); i ++) {
//		batchInputShape.push_back(inputShape[i]);
		trajInputShape.push_back(inputShape[i]);
	}

	std::srand(unsigned (std::time(0)));//TODO: To remove
}


template<typename NetType, typename EnvType, typename PolicyType>
void APPOWorker<NetType, EnvType, PolicyType>::train(const int updateNum) {
	torch::NoGradGuard guard;

	LOG4CXX_INFO(logger, "training: is multi " << dqnOption.multiLifes << " lives " << dqnOption.donePerEp);
	load();

	int updateIndex = 0;
	uint64_t stepNum = 0;
	int epCount = 0;

	std::vector<float> statRewards(dqnOption.envNum, 0);
	std::vector<float> statLens(dqnOption.envNum, 0);
	std::vector<float> sumRewards(dqnOption.envNum, 0);
	std::vector<float> sumLens(dqnOption.envNum, 0);
	std::vector<int> liveCounts(dqnOption.envNum, 0);

	std::vector<float> stateVec = env.reset();
	while (updateIndex < updateNum) {
//		LOG4CXX_INFO(logger, "---------------------------------------> update  " << updateIndex);

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
			stepNum ++;

			torch::Tensor stateTensor = torch::from_blob(stateVec.data(), inputShape).div(dqnOption.inputScale).to(deviceType);
			std::vector<torch::Tensor> rc = bModel.forward(stateTensor);
			valuesVec.push_back(rc[1]);
			pisVec.push_back(rc[0]);
			//probe test
//			LOG4CXX_INFO(logger, "state: " << stateTensor);
//			LOG4CXX_INFO(logger, "value: " << rc[1]);

			auto actionProbs =  torch::softmax(rc[0], -1);
			std::vector<int64_t> actions = policy.getActions(actionProbs);
//			LOG4CXX_INFO(logger, "action: " << actions);

			auto stepResult = env.step(actions, false);
			auto nextStateVec = std::get<0>(stepResult);
			auto rewardVec = std::get<1>(stepResult);
			auto doneVec = std::get<2>(stepResult);

			Stats::UpdateReward(statRewards, rewardVec);
			Stats::UpdateLen(statLens);


			std::vector<float> doneMaskVec(doneVec.size(), 1);
			for (int i = 0; i < doneVec.size(); i ++) {
				if (doneVec[i]) {
					doneMaskVec[i] = 0;

					tLogger.add_scalar("worker/len", stepNum, statLens[i]);
					tLogger.add_scalar("worker/reward", stepNum, statRewards[i]);
					LOG4CXX_INFO(logger, "" << stepNum << ": " << statLens[i] << ", " << statRewards[i]);

					if (dqnOption.multiLifes) {
						liveCounts[i] ++;
						sumLens[i] += statLens[i];
						sumRewards[i] += statRewards[i];

						if (liveCounts[i] >= dqnOption.donePerEp) {
							epCount ++;
							tLogger.add_scalar("worker/epLen", epCount, sumLens[i]);
							tLogger.add_scalar("worker/epReward", epCount, sumRewards[i]);
							LOG4CXX_INFO(logger, "Wrapper episode " << epCount << ": " << sumLens[i] << ", " << sumRewards[i]);

							sumLens[i] = 0;
							sumRewards[i] = 0;
							liveCounts[i] = 0;
						}
					}
					statLens[i] = 0;
					statRewards[i] = 0;
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
		torch::Tensor nextValueTensor = lastValueTensor.to(torch::kCPU);
		torch::Tensor gaeReturn = torch::zeros({dqnOption.envNum, 1});
		torch::Tensor plainReturn = torch::zeros({dqnOption.envNum, 1});
		plainReturn.copy_(nextValueTensor);
		LOG4CXX_DEBUG(logger, "nextValueTensor " << nextValueTensor.sizes());

		LOG4CXX_DEBUG(logger, "--------------------------------------> Calculate GAE");
		//tmp solution
		//TODO: Maybe there are functions as reduce
		torch::Tensor returnDoneTensor = torch::zeros(batchValueShape);
		returnDoneTensor.copy_(doneTensor);
		for (int i = dqnOption.trajStepNum - 1; i >= 0; i --) {
			torch::Tensor delta = rewardTensor[i] + dqnOption.gamma * nextValueTensor * doneTensor[i] - valueTensor[i];
			gaeReturn = delta + dqnOption.ppoLambda * dqnOption.gamma * gaeReturn * doneTensor[i];
//			LOG4CXX_INFO(logger, "Gae i " << gaeReturn);

			gaeReturns[i].copy_(gaeReturn);

			if (!dqnOption.tdValue) {
				plainReturn = rewardTensor[i] + dqnOption.gamma * plainReturn * doneTensor[i];
				returns[i].copy_(plainReturn);
			}
			nextValueTensor = valueTensor[i];
		}
		if (dqnOption.tdValue) {
			returns = gaeReturns + valueTensor; //from baseline3.
		}
//		float gaeMean = gaeReturns.mean().item<float>();
//		tLogger.add_scalar("worker/gae", updateIndex, gaeMean);

		//Put all tensors into CPU
//		valueTensor = valueTensor.detach().to(deviceType);
		LOG4CXX_DEBUG(logger, "Calculated GAE " << gaeReturns.sizes());

		//Calculate old log Pi
		torch::Tensor oldDistTensor = torch::stack(pisVec, 0).view({dqnOption.trajStepNum, dqnOption.envNum, actionNum});
		oldDistTensor = torch::softmax(oldDistTensor, -1).to(deviceType);
		LOG4CXX_DEBUG(logger, "oldDistTensor: " << oldDistTensor.sizes());

		auto actionData = EnvUtils::FlattenVector(actionsVec);
		torch::Tensor oldActionTensor = torch::from_blob(actionData.data(), {dqnOption.trajStepNum, dqnOption.envNum, 1}, longOpt).to(deviceType);
		LOG4CXX_DEBUG(logger, "oldActionTensor: " << oldActionTensor.sizes());
		torch::Tensor oldPiTensor = oldDistTensor.gather(-1, oldActionTensor).to(torch::kCPU);
		oldActionTensor = oldActionTensor.to(torch::kCPU);
		LOG4CXX_DEBUG(logger, "oldPiTensor: " << oldPiTensor.sizes());

		auto stateData = EnvUtils::FlattenVector(statesVec);
		auto stateTensor = torch::from_blob(stateData.data(), trajInputShape).div(dqnOption.inputScale);
		gaeReturns = gaeReturns.view({dqnOption.trajStepNum * dqnOption.envNum, 1}).detach();
		oldPiTensor = oldPiTensor.view({dqnOption.trajStepNum * dqnOption.envNum, 1}).detach();
		oldActionTensor = oldActionTensor.view({dqnOption.trajStepNum * dqnOption.envNum, 1});
		returns = returns.view({dqnOption.trajStepNum * dqnOption.envNum, 1}).detach(); //TODO: Moved to other thread impact NoGrad property?

		q.push({stateTensor, returns, gaeReturns, oldPiTensor, oldActionTensor});

	}

	save();
}

template<typename NetType, typename EnvType, typename PolicyType>
void APPOWorker<NetType, EnvType, PolicyType>::save() {
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
void APPOWorker<NetType, EnvType, PolicyType>::load() {
	if (!dqnOption.loadModel) {
		return;
	}

	std::string modelPath = dqnOption.loadPathPrefix + "_model.pt";
	torch::serialize::InputArchive inChive;
	inChive.load_from(modelPath);
	bModel.load(inChive);
	LOG4CXX_INFO(logger, "Load model from " << modelPath);

}


#endif /* INC_ALG_APPO_APPOWORKER_HPP_ */
