/*
 * algtester.hpp
 *
 *  Created on: Nov 22, 2021
 *      Author: zf
 */

#ifndef INC_ALG_ALGTESTER_HPP_
#define INC_ALG_ALGTESTER_HPP_

#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <tensorboard_logger.h>

#include <vector>

#include "dqnoption.h"
#include "gymtest/utils/stats.h"

template<typename NetType, typename EnvType, typename PolicyType>
class AlgTester {
private:
	NetType& net;
	EnvType& testEnv;
	PolicyType& policy;

	TensorBoardLogger tLogger;
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("algtester");

	const DqnOption dqnOption;

	torch::Tensor valueItems; //category dqn

	int testEpCount = 0;
	int testLifeCount = 0;

	//Gym server failed to reset by reset()
	std::vector<float> statRewards; //(dqnOption.testBatch, 0);
	std::vector<float> statLens; //(dqnOption.testBatch, 0);
	std::vector<float> sumRewards; //(dqnOption.testBatch, 0);
	std::vector<float> sumLens; //(dqnOption.testBatch, 0);
	std::vector<int> liveCounts; //(dqnOption.testBatch, 0);
	std::vector<int> noReward; //(dqnOption.testBatch, 0);
	std::vector<int> randomStep; //(dqnOption.testBatch, 0);

	void init();

public:
	AlgTester(NetType& iNet, EnvType& iEnv, PolicyType& iPolicy, const DqnOption& option);
	AlgTester(NetType& iNet, EnvType& iEnv, PolicyType& iPolicy, const DqnOption& option, TensorBoardLogger& tensorLogger);
	~AlgTester() = default;
	AlgTester(const AlgTester&) = delete;
	AlgTester operator=(const AlgTester&) = delete;

	void testAC();
	void testPlain();
	void testCategory();
};

template<typename NetType, typename EnvType, typename PolicyType>
AlgTester<NetType, EnvType, PolicyType>::AlgTester(NetType& iNet, EnvType& iEnv, PolicyType& iPolicy, const DqnOption& option):
	net(iNet),
	testEnv(iEnv),
	policy(iPolicy),
	dqnOption(option),
	tLogger(option.tensorboardLogPath.c_str())
{
	init();
}

template<typename NetType, typename EnvType, typename PolicyType>
AlgTester<NetType, EnvType, PolicyType>::AlgTester(NetType& iNet, EnvType& iEnv, PolicyType& iPolicy, const DqnOption& option, TensorBoardLogger& tensorLogger):
	net(iNet),
	testEnv(iEnv),
	policy(iPolicy),
	dqnOption(option),
	tLogger(tensorLogger)
{
	init();
}

template<typename NetType, typename EnvType, typename PolicyType>
void AlgTester<NetType, EnvType, PolicyType>::init() {
	statRewards = std::vector<float>(dqnOption.testBatch, 0);
	statLens = std::vector<float>(dqnOption.testBatch, 0);
	sumRewards = std::vector<float>(dqnOption.testBatch, 0);
	sumLens = std::vector<float>(dqnOption.testBatch, 0);
	liveCounts = std::vector<int>(dqnOption.testBatch, 0);
	noReward = std::vector<int>(dqnOption.testBatch, 0);
	randomStep = std::vector<int>(dqnOption.testBatch, 0);

	valueItems = torch::linspace(dqnOption.vMin, dqnOption.vMax, dqnOption.atomNum).to(dqnOption.deviceType);

}

template<typename NetType, typename EnvType, typename PolicyType>
void AlgTester<NetType, EnvType, PolicyType>::testAC() {
	LOG4CXX_INFO(logger, "To test " << dqnOption.testEp << "episodes");
	if (!dqnOption.toTest) {
		return;
	}

	int epCount = 0;

	torch::NoGradGuard guard;
	std::vector<float> states = testEnv.reset();
	while (epCount < dqnOption.testEp) {
		torch::Tensor stateTensor = torch::from_blob(states.data(), dqnOption.testInputShape).div(dqnOption.inputScale).to(dqnOption.deviceType);

		std::vector<torch::Tensor> rc = net.forward(stateTensor);
		auto actionOutput = rc[0];
		auto valueOutput = rc[1];
		auto actionProbs = torch::softmax(actionOutput, -1);
		//TODO: To replace by getActions

		std::vector<int64_t> actions = policy.getTestActions(actionProbs);
		if (dqnOption.randomHang) {
			for (int i = 0; i < dqnOption.testBatch; i ++) {
				if (noReward[i] >= dqnOption.hangNumTh) {
					actions[i] = torch::rand({1}).item<float>() * dqnOption.testOutput;
					LOG4CXX_INFO(logger, "random action for " << i << ": " << actions[i]);
					randomStep[i] ++;
					if (randomStep[i] > dqnOption.randomStep) {
						randomStep[i] = 0;
						noReward[i] = 0;
					}
				}
			}
		}
		auto stepResult = testEnv.step(actions, false);
		auto nextStateVec = std::get<0>(stepResult);
		auto rewardVec = std::get<1>(stepResult);
		auto doneVec = std::get<2>(stepResult);

		Stats::UpdateReward(statRewards, rewardVec);
		Stats::UpdateLen(statLens);

		for (int i = 0; i < dqnOption.testBatch; i ++) {
			if (doneVec[i]) {
				LOG4CXX_DEBUG(logger, "testEnv " << i << "done");
				epCount ++;
				testEpCount ++;

				sumRewards[i] += statRewards[i];
				sumLens[i] += statLens[i];

				LOG4CXX_INFO(logger, "test -----------> "<< i << " " << statLens[i] << ", " << statRewards[i]);
				tLogger.add_scalar("test/len", testEpCount, statLens[i]);
				tLogger.add_scalar("test/reward", testEpCount, statRewards[i]);
				statLens[i] = 0;
				statRewards[i] = 0;

				if (dqnOption.multiLifes) {
					liveCounts[i] ++;
					if (liveCounts[i] >= dqnOption.donePerEp) {
						LOG4CXX_INFO(logger, "TEST Wrapper episode " << i << " ----------------------------> " << sumRewards[i]);
						tLogger.add_scalar("test/sumlen", testEpCount, sumLens[i]);
						tLogger.add_scalar("test/sumreward", testEpCount, sumRewards[i]);

						liveCounts[i] = 0;
						sumRewards[i] = 0;
						sumLens[i] = 0;
					}
				}
			}

			if (dqnOption.randomHang) {
			//Good action should have reward
				if (rewardVec[i] < dqnOption.hangRewardTh) { //for float compare
					noReward[i] ++;
				}
			}
		}
		states = nextStateVec;
	}
}

//TODO: summarize them by template condition
template<typename NetType, typename EnvType, typename PolicyType>
void AlgTester<NetType, EnvType, PolicyType>::testPlain() {
	int epCount = 0;

	torch::NoGradGuard guard;
	std::vector<float> states = testEnv.reset();
	while (epCount < dqnOption.testEp) {
		torch::Tensor stateTensor = torch::from_blob(states.data(), dqnOption.testInputShape).div(dqnOption.inputScale).to(dqnOption.deviceType);

		torch::Tensor actionOutput = net.forward(stateTensor);
		std::vector<int64_t> actions = policy.getTestActions(actionOutput);

		if (dqnOption.randomHang) {
			for (int i = 0; i < dqnOption.testBatch; i ++) {
				if (noReward[i] >= dqnOption.hangNumTh) {
					actions[i] = torch::rand({1}).item<float>() * dqnOption.testOutput;
					LOG4CXX_INFO(logger, "random action for " << i << ": " << actions[i]);
					randomStep[i] ++;
					if (randomStep[i] > dqnOption.randomStep) {
						randomStep[i] = 0;
						noReward[i] = 0;
					}
				}
			}
		}
		auto stepResult = testEnv.step(actions, false);
		auto nextStateVec = std::get<0>(stepResult);
		auto rewardVec = std::get<1>(stepResult);
		auto doneVec = std::get<2>(stepResult);

		Stats::UpdateReward(statRewards, rewardVec);
		Stats::UpdateLen(statLens);

		for (int i = 0; i < dqnOption.testBatch; i ++) {
			if (doneVec[i]) {
				LOG4CXX_DEBUG(logger, "testEnv " << i << "done");
//				epCount ++;
				testLifeCount ++;

				LOG4CXX_INFO(logger, "test -----------> "<< i << " " << statLens[i] << ", " << statRewards[i]);
				tLogger.add_scalar("test/len", testLifeCount, statLens[i]);
				tLogger.add_scalar("test/reward", testLifeCount, statRewards[i]);


				if (dqnOption.multiLifes) {
					liveCounts[i] ++;
					sumRewards[i] += statRewards[i];
					sumLens[i] += statLens[i];

					if (liveCounts[i] >= dqnOption.donePerEp) {
						epCount ++;
						testEpCount ++;

						LOG4CXX_INFO(logger, "TEST Wrapper episode " << i << " ----------------------------> " << sumRewards[i]);

						tLogger.add_scalar("test/sumlen", testEpCount, sumLens[i]);
						tLogger.add_scalar("test/sumreward", testEpCount, sumRewards[i]);
						liveCounts[i] = 0;
						sumRewards[i] = 0;
						sumLens[i] = 0;
					}
				} else {
					epCount ++;
					testEpCount ++;
				}

				statLens[i] = 0;
				statRewards[i] = 0;
			}

			if (dqnOption.randomHang) {
			//Good action should have reward
				if (rewardVec[i] < dqnOption.hangRewardTh) { //for float compare
					noReward[i] ++;
				}
			}
		}
		states = nextStateVec;
	}

}

template<typename NetType, typename EnvType, typename PolicyType>
void AlgTester<NetType, EnvType, PolicyType>::testCategory() {
	int epCount = 0;

	torch::NoGradGuard guard;
	std::vector<float> states = testEnv.reset();
	while (epCount < dqnOption.testEp) {
		torch::Tensor stateTensor = torch::from_blob(states.data(), dqnOption.testInputShape).div(dqnOption.inputScale).to(dqnOption.deviceType);

		torch::Tensor actionOutput = net.forward(stateTensor);
		actionOutput = actionOutput.view({dqnOption.testBatch, dqnOption.outputNum, dqnOption.atomNum}).softmax(-1);
		auto qs = (actionOutput * valueItems).sum(-1, false);
		std::vector<int64_t> actions = policy.getTestActions(qs);

		if (dqnOption.randomHang) {
			for (int i = 0; i < dqnOption.testBatch; i ++) {
				if (noReward[i] >= dqnOption.hangNumTh) {
					actions[i] = torch::rand({1}).item<float>() * dqnOption.testOutput;
					LOG4CXX_INFO(logger, "random action for " << i << ": " << actions[i]);
					randomStep[i] ++;
					if (randomStep[i] > dqnOption.randomStep) {
						randomStep[i] = 0;
						noReward[i] = 0;
					}
				}
			}
		}
		auto stepResult = testEnv.step(actions, false);
		auto nextStateVec = std::get<0>(stepResult);
		auto rewardVec = std::get<1>(stepResult);
		auto doneVec = std::get<2>(stepResult);

		Stats::UpdateReward(statRewards, rewardVec);
		Stats::UpdateLen(statLens);

		for (int i = 0; i < dqnOption.testBatch; i ++) {
			if (doneVec[i]) {
				LOG4CXX_DEBUG(logger, "testEnv " << i << "done");
//				epCount ++;
				testLifeCount ++;

				LOG4CXX_INFO(logger, "test -----------> "<< i << " " << statLens[i] << ", " << statRewards[i]);
				tLogger.add_scalar("test/len", testLifeCount, statLens[i]);
				tLogger.add_scalar("test/reward", testLifeCount, statRewards[i]);


				if (dqnOption.multiLifes) {
					liveCounts[i] ++;
					sumRewards[i] += statRewards[i];
					sumLens[i] += statLens[i];

					if (liveCounts[i] >= dqnOption.donePerEp) {
						epCount ++;
						testEpCount ++;

						LOG4CXX_INFO(logger, "TEST Wrapper episode " << i << " ----------------------------> " << sumRewards[i]);

						tLogger.add_scalar("test/sumlen", testEpCount, sumLens[i]);
						tLogger.add_scalar("test/sumreward", testEpCount, sumRewards[i]);
						liveCounts[i] = 0;
						sumRewards[i] = 0;
						sumLens[i] = 0;
					}
				} else {
					epCount ++;
					testEpCount ++;
				}

				statLens[i] = 0;
				statRewards[i] = 0;
			}

			if (dqnOption.randomHang) {
			//Good action should have reward
				if (rewardVec[i] < dqnOption.hangRewardTh) { //for float compare
					noReward[i] ++;
				}
			}
		}
		states = nextStateVec;
	}

}
#endif /* INC_ALG_ALGTESTER_HPP_ */
