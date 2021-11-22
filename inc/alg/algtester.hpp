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

	int testEpCount = 0;

public:
	AlgTester(NetType& iNet, EnvType& iEnv, PolicyType& iPolicy, const DqnOption& option);
	~AlgTester() = default;
	AlgTester(const AlgTester&) = delete;
	AlgTester operator=(const AlgTester&) = delete;

	void test();
};

template<typename NetType, typename EnvType, typename PolicyType>
AlgTester<NetType, EnvType, PolicyType>::AlgTester(NetType& iNet, EnvType& iEnv, PolicyType& iPolicy, const DqnOption& option):
	net(iNet),
	testEnv(iEnv),
	policy(iPolicy),
	dqnOption(option),
	tLogger(option.tensorboardLogPath.c_str())
{

}

template<typename NetType, typename EnvType, typename PolicyType>
void AlgTester<NetType, EnvType, PolicyType>::test() {
	LOG4CXX_INFO(logger, "To test " << dqnOption.testEp << "episodes");
	if (!dqnOption.toTest) {
		return;
	}

	int epCount = 0;
	std::vector<float> statRewards(dqnOption.testBatch, 0);
	std::vector<float> statLens(dqnOption.testBatch, 0);
	std::vector<float> sumRewards(dqnOption.testBatch, 0);
	std::vector<float> sumLens(dqnOption.testBatch, 0);
	std::vector<int> liveCounts(dqnOption.testBatch, 0);
	std::vector<int> noReward(dqnOption.testBatch, 0);
	std::vector<int> randomStep(dqnOption.testBatch, 0);

	torch::NoGradGuard guard;
	std::vector<float> states = testEnv.reset();
	while (epCount < dqnOption.testEp) {
//		torch::Tensor stateTensor = torch::from_blob(states.data(), inputShape).to(deviceType);
		torch::Tensor stateTensor = torch::from_blob(states.data(), dqnOption.inputShape).div(dqnOption.inputScale).to(dqnOption.deviceType);

		std::vector<torch::Tensor> rc = net.forward(stateTensor);
		auto actionOutput = rc[0]; //TODO: detach?
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
//				auto resetResult = env.reset(i);
				//udpate nextstatevec, target mask
//				std::copy(resetResult.begin(), resetResult.end(), nextStateVec.begin() + (offset * i));
				epCount ++;
				testEpCount ++;

				sumRewards[i] += statRewards[i];
				sumLens[i] += statLens[i];

				LOG4CXX_INFO(logger, "test -----------> "<< i << " " << statLens[i] << ", " << statRewards[i]);
				tLogger.add_scalar("test/len", testEpCount, statLens[i]);
				tLogger.add_scalar("test/reward", testEpCount, statRewards[i]);
//				testStater.update(statLens[i], statRewards[i]);
				statLens[i] = 0;
				statRewards[i] = 0;
//				stater.printCurStat();

				liveCounts[i] ++;
				if (liveCounts[i] >= dqnOption.donePerEp) {
					LOG4CXX_INFO(logger, "Wrapper episode " << i << " ----------------------------> " << sumRewards[i]);
//					sumStater.update(sumLens[i], sumRewards[i]);
					tLogger.add_scalar("test/sumlen", testEpCount, sumLens[i]);
					tLogger.add_scalar("test/sumreward", testEpCount, sumRewards[i]);
					liveCounts[i] = 0;
					sumRewards[i] = 0;
					sumLens[i] = 0;
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


#endif /* INC_ALG_ALGTESTER_HPP_ */
