/*
 * nbrbdqn.hpp
 *
 *  Created on: Apr 11, 2021
 *      Author: zf
 */

#ifndef INC_ALG_NBRBDQN_HPP_
#define INC_ALG_NBRBDQN_HPP_

#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <vector>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/nobatchrb.h"
#include "gymtest/utils/stats.h"
#include "dqnoption.h"


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
class NbRbDqn {
private:
	NetType& model;
	EnvType& env;
	EnvType& testEnv;
	PolicyType& policy;
	OptimizerType& optimizer;
	const torch::Device deviceType;
	const at::IntArrayRef inputShape;

	NoBatchRB rb;
	Stats stater;
	Stats testStater;
	int offset = 0;

	const torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	torch::nn::SmoothL1Loss lossComputer = torch::nn::SmoothL1Loss();

	log4cxx::LoggerPtr logger;

	void update(const int batchSize, const int epochNum);
	void test(const int batchSize, const int epochNum);
public:
	float gamma = 0.99;
	const int testEp = 16;

	NbRbDqn(NetType& iModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, DqnOption option);
	~NbRbDqn() = default;

	void train(const int batchSize, const int epochNum);
};

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
NbRbDqn<NetType, EnvType, PolicyType, OptimizerType>::NbRbDqn(
		NetType& iModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer,
		DqnOption option):
	model(iModel),
	env(iEnv),
	testEnv(tEnv),
	policy(iPolicy),
	optimizer(iOptimizer),
	deviceType(option.deviceType),
	inputShape(option.inputShape),
	rb(option.rbCap),
	stater(option.statPath),
	testStater("./test_stat.txt"),
	logger(log4cxx::Logger::getLogger("dqn"))
{
	//model and optimizer set device type before this constructor
	offset = 1;
	for (int i = 1; i < inputShape.size(); i ++) {
		offset *= inputShape[i];
	}
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void NbRbDqn<NetType, EnvType, PolicyType, OptimizerType>::update(const int batchSize, const int epochNum) {
	std::vector<int64_t> stateShape(inputShape.size(), 0);
	stateShape[0] = 1;
	for (int i = 1; i < stateShape.size(); i ++) {
		stateShape[i] = inputShape[i];
	}
	model.train();

	for (int i = 0; i < epochNum; i ++) {
		std::vector<torch::Tensor> subStates;
		std::vector<torch::Tensor> subNextStates;
		std::vector<float> subRewards;
		std::vector<long> subActions;
		std::vector<bool> subDones;

		for (int j = 0; j < batchSize; j ++) {
			int index = rb.randSelect();

			//TODO: guarantee capacity to avoid repeat
			subStates.push_back(torch::from_blob(rb.states[index].data(), stateShape).to(deviceType));
			subNextStates.push_back(torch::from_blob(rb.nextStates[index].data(), stateShape).to(deviceType));

			subRewards.push_back(rb.rewards[index]);
			subActions.push_back(rb.actions[index]);
			subDones.push_back(rb.dones[index]);
		}
		torch::Tensor stateTensor = torch::cat(subStates, 0);
		torch::Tensor nextStateTensor = torch::cat(subNextStates, 0);
		torch::Tensor rewardTensor = torch::from_blob(subRewards.data(), {batchSize, 1}).to(deviceType);
		torch::Tensor actionTensor = torch::from_blob(subActions.data(), {batchSize, 1}, longOpt).to(deviceType);

		torch::Tensor targetMask = torch::ones({batchSize, 1}).to(deviceType);
		for (int j = 0; j < batchSize; j ++) {
			if (subDones[j]) {
				targetMask[j][0] = 0;
			}
		}

		torch::Tensor outputTensor = model.forward(stateTensor);
		torch::Tensor nextOutputTensor = model.forward(nextStateTensor).detach();
		torch::Tensor lossInput = outputTensor.gather(-1, actionTensor);
		torch::Tensor target = std::get<0>(nextOutputTensor.max(-1)).unsqueeze(-1) * gamma;
		target = target * targetMask + rewardTensor;

		torch::Tensor loss = lossComputer(lossInput, target);
		optimizer.zero_grad();
		loss.backward();
		torch::nn::utils::clip_grad_value_(optimizer.parameters(), 1);
		optimizer.step();
	}
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void NbRbDqn<NetType, EnvType, PolicyType, OptimizerType>::train(const int batchSize, const int epochNum) {
	const int stepGap = rb.capacity / batchSize;
	int step = 0;
	int epCount = 0;
	auto lossComputer = torch::nn::SmoothL1Loss();
	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);

	std::vector<float> statRewards(batchSize, 0);
	std::vector<float> statLens(batchSize, 0);

	model.eval();
	std::vector<float> stateVec = env.reset();
	while (epCount < epochNum) {
		torch::Tensor inputTensor = torch::from_blob(stateVec.data(), inputShape).to(deviceType);
		torch::Tensor outputTensor = model.forward(inputTensor);
		std::vector<int64_t> actions = policy.getActions(outputTensor);

		auto stepResult = env.step(actions);
		auto nextStateVec = std::get<0>(stepResult);
		auto rewardVec = std::get<1>(stepResult);
		auto doneVec = std::get<2>(stepResult);

		Stats::UpdateReward(statRewards, rewardVec);
		Stats::UpdateLen(statLens);

//    	auto targetMask = torch::ones({clientNum, 1}).to(deviceType);
		for (int i = 0; i < doneVec.size(); i ++) {
			if (doneVec[i]) {
				LOG4CXX_DEBUG(logger, "env " << i << "done");
				auto resetResult = env.reset(i);
				//udpate nextstatevec, target mask
				std::copy(resetResult.begin(), resetResult.end(), nextStateVec.begin() + (offset * i));
				epCount ++;

				stater.update(statLens[i], statRewards[i]);
				statLens[i] = 0;
				statRewards[i] = 0;
//				stater.printCurStat();
				LOG4CXX_INFO(logger, stater);
			}
		}

		rb.put(stateVec, nextStateVec, rewardVec, actions, doneVec, batchSize);
    	stateVec = nextStateVec;

		step ++;
		if (step >= stepGap) {
			LOG4CXX_INFO(logger, "---------------------> To update");
			step = 0;

			//TODO: Check epochNum
			update(batchSize, stepGap);
			test(batchSize, testEp);
			model.eval();
		}
	}
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void NbRbDqn<NetType, EnvType, PolicyType, OptimizerType>::test(const int batchSize, const int epochNum) {
	model.eval();

	int epCount = 0;
	std::vector<float> statRewards(batchSize, 0);
	std::vector<float> statLens(batchSize, 0);

	std::vector<float> stateVec = testEnv.reset();
	while (epCount < epochNum) {
		torch::Tensor inputTensor = torch::from_blob(stateVec.data(), inputShape).to(deviceType);
		torch::Tensor outputTensor = model.forward(inputTensor);
		std::vector<int64_t> actions = policy.getTestActions(outputTensor);
		LOG4CXX_INFO(logger, "test actions: " << actions);

		auto stepResult = testEnv.step(actions);
		auto nextStateVec = std::get<0>(stepResult);
		auto rewardVec = std::get<1>(stepResult);
		auto doneVec = std::get<2>(stepResult);

		Stats::UpdateReward(statRewards, rewardVec);
		Stats::UpdateLen(statLens);

//    	auto targetMask = torch::ones({clientNum, 1}).to(deviceType);
		for (int i = 0; i < doneVec.size(); i ++) {
			if (doneVec[i]) {
				LOG4CXX_DEBUG(logger, "env " << i << "done");
				auto resetResult = testEnv.reset(i);
				//udpate nextstatevec, target mask
				std::copy(resetResult.begin(), resetResult.end(), nextStateVec.begin() + (offset * i));
				epCount ++;

				testStater.update(statLens[i], statRewards[i]);
				statLens[i] = 0;
				statRewards[i] = 0;
//				stater.printCurStat();
				LOG4CXX_INFO(logger, "test_" << testStater);
			}
		}

    	stateVec = nextStateVec;
	}

//	model.train();
}



#endif /* INC_ALG_NBRBDQN_HPP_ */
