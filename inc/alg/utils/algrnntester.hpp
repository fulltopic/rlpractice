/*
 * algrnntester.hpp
 *
 *  Created on: Jan 31, 2022
 *      Author: zf
 */

#ifndef INC_ALG_UTILS_ALGRNNTESTER_HPP_
#define INC_ALG_UTILS_ALGRNNTESTER_HPP_

#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <tensorboard_logger.h>

#include <vector>

#include "dqnoption.h"
#include "gymtest/utils/stats.h"

template<typename NetType, typename EnvType, typename PolicyType>
class AlgRNNTester {
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

	std::vector<torch::Tensor> stepStates;
	std::vector<int64_t> stepInputShape;


	void init();

public:
	AlgRNNTester(NetType& iNet, EnvType& iEnv, PolicyType& iPolicy, const DqnOption& option);
	AlgRNNTester(NetType& iNet, EnvType& iEnv, PolicyType& iPolicy, const DqnOption& option, TensorBoardLogger& tensorLogger);
	~AlgRNNTester() = default;
	AlgRNNTester(const AlgRNNTester&) = delete;
	AlgRNNTester operator=(const AlgRNNTester&) = delete;

	void testAC();
//	void testPlain();
//	void testCategory();

//	std::vector<torch::Tensor> testACRNN(std::vector<torch::Tensor>& hiddenState);
};

template<typename NetType, typename EnvType, typename PolicyType>
AlgRNNTester<NetType, EnvType, PolicyType>::AlgRNNTester(NetType& iNet, EnvType& iEnv, PolicyType& iPolicy, const DqnOption& option):
	net(iNet),
	testEnv(iEnv),
	policy(iPolicy),
	dqnOption(option),
	tLogger(option.tensorboardLogPath.c_str())
{
	init();
}

template<typename NetType, typename EnvType, typename PolicyType>
AlgRNNTester<NetType, EnvType, PolicyType>::AlgRNNTester(NetType& iNet, EnvType& iEnv, PolicyType& iPolicy, const DqnOption& option, TensorBoardLogger& tensorLogger):
	net(iNet),
	testEnv(iEnv),
	policy(iPolicy),
	dqnOption(option),
	tLogger(tensorLogger)
{
	init();
}

template<typename NetType, typename EnvType, typename PolicyType>
void AlgRNNTester<NetType, EnvType, PolicyType>::init() {
	statRewards = std::vector<float>(dqnOption.testBatch, 0);
	statLens = std::vector<float>(dqnOption.testBatch, 0);
	sumRewards = std::vector<float>(dqnOption.testBatch, 0);
	sumLens = std::vector<float>(dqnOption.testBatch, 0);
	liveCounts = std::vector<int>(dqnOption.testBatch, 0);
	noReward = std::vector<int>(dqnOption.testBatch, 0);
	randomStep = std::vector<int>(dqnOption.testBatch, 0);

	valueItems = torch::linspace(dqnOption.vMin, dqnOption.vMax, dqnOption.atomNum).to(dqnOption.deviceType);

	for (int i = 0; i < dqnOption.hiddenNums.size(); i ++) {
		stepStates.push_back(torch::zeros({
			dqnOption.hidenLayerNums[i], dqnOption.testBatch, dqnOption.hiddenNums[i]
		}).to(dqnOption.deviceType));
	}

	stepInputShape.push_back(dqnOption.testBatch);
	stepInputShape.push_back(1);
	int dataNum = 1;
	for (int i = 1; i < dqnOption.testInputShape.size(); i ++) {
		dataNum *= dqnOption.testInputShape[i];
	}
	stepInputShape.push_back(dataNum);
}

template<typename NetType, typename EnvType, typename PolicyType>
void AlgRNNTester<NetType, EnvType, PolicyType>::testAC() {
	LOG4CXX_INFO(logger, "To test " << dqnOption.testEp << " episodes");
	if (!dqnOption.toTest) {
		return;
	}

	int epCount = 0;

	torch::NoGradGuard guard;
	std::vector<float> states = testEnv.reset();
	while (epCount < dqnOption.testEp) {
		torch::Tensor stateTensor = torch::from_blob(states.data(), stepInputShape).div(dqnOption.inputScale).to(dqnOption.deviceType);
//		stateTensor = stateTensor.narrow(2, 0, 3); //TODO: tmp solution

		std::vector<torch::Tensor> rc = net.forward(stateTensor, stepStates);
		auto actionOutput = rc[0];
		auto valueOutput = rc[1];
		auto actionProbs = torch::softmax(actionOutput, -1);
		//TODO: To replace by getActions

		std::vector<int64_t> actions = policy.getTestActions(actionProbs);

		auto stepResult = testEnv.step(actions, false);
		auto nextStateVec = std::get<0>(stepResult);
		auto rewardVec = std::get<1>(stepResult);
		auto doneVec = std::get<2>(stepResult);

		Stats::UpdateReward(statRewards, rewardVec);
		Stats::UpdateLen(statLens);
		for (int i = 0; i < dqnOption.testBatch; i ++) {
			if (doneVec[i]) {
				LOG4CXX_DEBUG(logger, "testEnv " << i << "done");

				for (int j = 0; j < dqnOption.hidenLayerNums.size(); j ++) {
					stepStates[j].fill_(0); //batch first
				}

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

	return;
}

#endif /* INC_ALG_UTILS_ALGRNNTESTER_HPP_ */
