/*
 * dqn.hpp
 *
 *  Created on: Apr 7, 2021
 *      Author: zf
 */

#ifndef INC_ALG_DQN_HPP_
#define INC_ALG_DQN_HPP_

#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <vector>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/replaybuffer.h"
#include "gymtest/utils/stats.h"
#include "dqnoption.h"

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
class Dqn {
private:
	NetType& model;
	EnvType& env;
	EnvType& testEnv;
	PolicyType& policy;
	OptimizerType& optimizer;
	const torch::Device deviceType;
	const at::IntArrayRef inputShape;

	ReplayBuffer rb;
	Stats stater;
	Stats testStater;
	int offset = 0;

	log4cxx::LoggerPtr logger;

	void test(const int batchSize, const int epochNum);
public:
	const float gamma;
	const int testEp = 16;

	//TODO: Seemed model is of shared_ptr, pass by value is OK
	Dqn(NetType& iModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, DqnOption option);
	~Dqn() = default;

	void train(const int batchSize, const int epochNum);
};

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
Dqn<NetType, EnvType, PolicyType, OptimizerType>::Dqn(NetType& iModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer,
		DqnOption option):
	model(iModel),
	env(iEnv),
	testEnv(tEnv),
	policy(iPolicy),
	optimizer(iOptimizer),
	deviceType(option.deviceType),
	inputShape(option.inputShape),
	rb(option.rbCap),
	gamma(option.gamma),
	stater(option.statPath),
	testStater("./test_stat.txt"),
	logger(log4cxx::Logger::getLogger("dqn"))
{
	//model and optimizer set device type before this constructor
//	model.to(deviceType);
	offset = 1;
	for (int i = 1; i < inputShape.size(); i ++) {
		offset *= inputShape[i];
	}
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void Dqn<NetType, EnvType, PolicyType, OptimizerType>::train(const int batchSize, const int epochNum) {
	const int stepGap = rb.capacity;
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

		rb.put(stateVec, nextStateVec, rewardVec, actions, doneVec);
    	stateVec = nextStateVec;

		step ++;
//		LOG4CXX_INFO(logger, "step: " << step);
		if (step >= stepGap) {
			LOG4CXX_INFO(logger, "---------------------> To update");
			model.train();
			step = 0;
			for (int i = 0; i < rb.capacity; i ++) {
				LOG4CXX_DEBUG(logger, "sample " << i);
				int index = rb.randSelect();
				LOG4CXX_DEBUG(logger, "index " << index);

				torch::Tensor stateTensor = torch::from_blob(rb.states[index].data(), inputShape).to(deviceType);
				LOG4CXX_DEBUG(logger, "created state tensor");
				torch::Tensor nextStateTensor = torch::from_blob(rb.nextStates[index].data(), inputShape).to(deviceType);
				torch::Tensor rewardTensor = torch::from_blob(rb.rewards[index].data(), {inputShape[0], 1}).to(deviceType);
				torch::Tensor actionTensor = torch::from_blob(rb.actions[index].data(), {inputShape[0], 1}, longOpt).to(deviceType);
				LOG4CXX_DEBUG(logger, "created action tensor");
		    	auto targetMask = torch::ones({inputShape[0], 1}).to(deviceType);
		    	for (int j = 0; j < rb.dones[index].size(); j ++) {
		    		if (rb.dones[index][j]) {
		    			targetMask[j][0] = 0;
		    		}
		    	}

		    	torch::Tensor outputTensor = model.forward(stateTensor);
		    	torch::Tensor nextOutputTensor = model.forward(nextStateTensor).detach();
		    	torch::Tensor lossInput = outputTensor.gather(-1, actionTensor);
		    	torch::Tensor target = std::get<0>(nextOutputTensor.max(-1)).unsqueeze(-1) * gamma;
		    	target = target * targetMask + rewardTensor;

		    	auto loss = lossComputer(lossInput, target);
		    	optimizer.zero_grad();
		    	loss.backward();
//		    	torch::nn::utils::clip_grad_value_(optimizer.parameters(), 1);
		    	optimizer.step();
			}

			test(batchSize, testEp);
		}
		model.eval();
	}
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void Dqn<NetType, EnvType, PolicyType, OptimizerType>::test(const int batchSize, const int epochNum) {
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
#endif /* INC_ALG_DQN_HPP_ */
