/*
 * envstep.hpp
 *
 *  Created on: Jan 20, 2022
 *      Author: zf
 */

#ifndef INC_ALG_ENVSTEP_HPP_
#define INC_ALG_ENVSTEP_HPP_


#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <tensorboard_logger.h>

#include <vector>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/stats.h"
#include "alg/utils/dqnoption.h"
#include "alg/utils/algtester.hpp"
#include "alg/utils/utils.hpp"

template <typename NetType, typename EnvType, typename PolicyType>
class A2CStoreEnvStep {
public:
	std::vector<std::vector<float>> statesVec;
	std::vector<std::vector<float>> rewardsVec;
	std::vector<std::vector<float>> donesVec;
	std::vector<std::vector<long>> actionsVec;
	std::vector<torch::Tensor> valuesVec;
	std::vector<torch::Tensor> pisVec;

	std::vector<float> stateVec;

	std::vector<float> statRewards; //(batchSize, 0);
	std::vector<float> statLens; //(batchSize, 0);
	std::vector<int> liveCounts; //(batchSize, 0);
	std::vector<float> sumRewards; //(batchSize, 0);
	std::vector<float> sumLens; //(batchSize, 0);

	int epCount = 0;
	int roundCount = 0;

	const DqnOption stepOption;
	const torch::Device deviceType;
//	const at::IntArrayRef inputShape;

	TensorBoardLogger tLogger;
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("envsteplog");

	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);

	A2CStoreEnvStep(const A2CStoreEnvStep&) = delete;

	//TODO: stateVec = env.reset
	A2CStoreEnvStep (DqnOption iOption): stepOption(iOption),
			deviceType(iOption.deviceType),
//			inputShape(iOption.inputShape),
			tLogger(iOption.tensorboardLogPath.c_str())
	{
		statRewards = std::vector<float>(stepOption.envNum, 0);
		statLens = std::vector<float>(stepOption.envNum, 0); //(batchSize, 0);
		liveCounts = std::vector<int>(stepOption.envNum, 0);
		sumRewards = std::vector<float>(stepOption.envNum, 0);
		sumLens = std::vector<float>(stepOption.envNum, 0);
	}

	void steps(NetType& bModel, EnvType& env, PolicyType& policy, const int maxStep, int& updateNum) {
		statesVec.clear();
		rewardsVec.clear();
		donesVec.clear();
		actionsVec.clear();
		valuesVec.clear();
		pisVec.clear();

		LOG4CXX_DEBUG(logger, "steps " << maxStep);

		torch::NoGradGuard guard;

		for (int step = 0; step < maxStep; step ++) {
			updateNum ++;

			torch::Tensor stateTensor = torch::from_blob(stateVec.data(), stepOption.inputShape).div(stepOption.inputScale).to(deviceType);
			std::vector<torch::Tensor> rc = bModel.forward(stateTensor);

			valuesVec.push_back(rc[1]);
			pisVec.push_back(rc[0]); //TODO: softmax
			LOG4CXX_DEBUG(logger, "valuesVec " << step << " " << valuesVec);

			auto actionProbs = torch::softmax(rc[0], -1);
			std::vector<int64_t> actions = policy.getActions(actionProbs);

			auto stepResult = env.step(actions,false);
			auto nextStateVec = std::get<0>(stepResult);
			auto rewardVec = std::get<1>(stepResult);
			auto doneVec = std::get<2>(stepResult);

			Stats::UpdateReward(statRewards, rewardVec);
			Stats::UpdateLen(statLens);

			std::vector<float> doneMaskVec(doneVec.size(), 1);
			for (int i = 0; i < doneVec.size(); i ++) {
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

					if (stepOption.multiLifes) {
						liveCounts[i] ++;
						if (liveCounts[i] >= stepOption.donePerEp) {
							roundCount ++;
							LOG4CXX_INFO(logger, "Wrapper episode " << i << "-----------------------------> " << sumRewards[i]);
							tLogger.add_scalar("train/sumLen", roundCount, sumLens[i]);
							tLogger.add_scalar("train/sumReward", roundCount, sumRewards[i]);

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

			stateVec = nextStateVec; //TODO: content destructed out of function?
		}
	}
};


#endif /* INC_ALG_ENVSTEP_HPP_ */
