/*
 * dqntarget.hpp
 *
 *  Created on: Apr 12, 2021
 *      Author: zf
 */

//deprecated
#ifndef INC_ALG_DQNTARGET_HPP_
#define INC_ALG_DQNTARGET_HPP_


#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <iostream>
#include <vector>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/replaybuffer.h"
#include "gymtest/utils/stats.h"
#include "dqnoption.h"

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
class DqnTarget {
private:
	NetType& model;
	NetType& targetModel;
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

	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("dqntarget");
	torch::nn::SmoothL1Loss lossComputer = torch::nn::SmoothL1Loss();
	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);

	std::vector<float> input1;
	std::vector<int64_t> action1;
	torch::Tensor output1;

	uint32_t updateNum = 0;
	const uint32_t UpdateExploreGap = 16;

	void update(const int batchSize);
	void test(const int batchSize, const int epochNum);

	void updateTarget();
public:
	const float gamma;
	const int testEp = 16;

	DqnTarget(NetType& iModel, NetType& tModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, DqnOption option);
	~DqnTarget() = default;

	void train(const int batchSize, const int epochNum);
};

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
DqnTarget<NetType, EnvType, PolicyType, OptimizerType>::DqnTarget(NetType& iModel, NetType& tModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer,
		DqnOption option):
	model(iModel),
	targetModel(tModel),
	env(iEnv),
	testEnv(tEnv),
	policy(iPolicy),
	optimizer(iOptimizer),
	deviceType(option.deviceType),
	inputShape(option.inputShape),
	rb(option.rbCap),
	gamma(option.gamma),
	stater(option.statPath),
	testStater("./test_stat.txt")
{
	//model and optimizer set device type before this constructor
	offset = 1;
	for (int i = 1; i < inputShape.size(); i ++) {
		offset *= inputShape[i];
	}

	targetModel.eval();
	updateTarget();
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void DqnTarget<NetType, EnvType, PolicyType, OptimizerType>::updateTarget() {
	LOG4CXX_INFO(logger, "Update target network");

	auto paramDict = model.named_parameters();
	auto buffDict = model.named_buffers();
	auto targetParamDict = targetModel.named_parameters();
	auto targetBuffDict = targetModel.named_buffers();

	for(const auto& item: paramDict) {
		const auto& key = item.key();
		const auto param = item.value();
		auto& targetParam = targetParamDict[key];
		{
			torch::NoGradGuard gd;
			targetParam.copy_(param);
		}
	}

	for (const auto& item: buffDict) {
		const auto& key = item.key();
		const auto& buff = item.value();
		auto& targetBuff = targetBuffDict[key];
		{
			torch::NoGradGuard gd;
			targetBuff.copy_(buff);
		}
	}
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void DqnTarget<NetType, EnvType, PolicyType, OptimizerType>::update(const int batchSize) {
	LOG4CXX_INFO(logger, "---------------------> To update");
//	model.train();
//	model.eval();

	for (int i = 0; i < rb.capacity; i ++) {
		LOG4CXX_DEBUG(logger, "sample " << i);
		int index = rb.randSelect();
//		int index = i;
		LOG4CXX_DEBUG(logger, "index " << index);
//		LOG4CXX_INFO(logger, "action vec: " << rb.actions[index]);
		torch::Tensor stateTensor = torch::from_blob(rb.states[index].data(), inputShape).to(deviceType);
		LOG4CXX_DEBUG(logger, "created state tensor");
		torch::Tensor nextStateTensor = torch::from_blob(rb.nextStates[index].data(), inputShape).to(deviceType);
		torch::Tensor rewardTensor = torch::from_blob(rb.rewards[index].data(), {inputShape[0], 1}).to(deviceType);
		torch::Tensor actionTensor = torch::from_blob(rb.actions[index].data(), {inputShape[0], 1}, longOpt).to(deviceType);
//		std::cout << "action vec: " << std::endl;
//		for (auto& a: rb.actions[index]){
//			std::cout << a << ", ";
//		}
//		std::cout << std::endl;
//		LOG4CXX_INFO(logger, "action tensor: " << actionTensor);
		LOG4CXX_DEBUG(logger, "created action tensor");
		auto targetMask = torch::ones({inputShape[0], 1}).to(deviceType);
		for (int j = 0; j < rb.dones[index].size(); j ++) {
		    if (rb.dones[index][j]) {
		    	targetMask[j][0] = 0;
		    }
		}

		torch::Tensor outputTensor = model.forward(stateTensor);
		torch::Tensor nextOutputTensor;
		{
			torch::NoGradGuard();
			nextOutputTensor = targetModel.forward(nextStateTensor).detach();
		}
		torch::Tensor lossInput = outputTensor.gather(-1, actionTensor);
//		if (index == 0) {
//			LOG4CXX_INFO(logger, "action index1: " << actionTensor);
//			LOG4CXX_INFO(logger, "outputTensor index1" << outputTensor);
//			LOG4CXX_INFO(logger, "lossInput index1: " << lossInput);
//			int diff = 0;
//			for (int j = 0; j < input1.size(); j ++) {
//				if (input1[j] != rb.states[index][j]) {
//					LOG4CXX_INFO(logger, "mismatch " << j << ": " << input1[j] << ", " << rb.states[index][j]);
//					diff ++;
//				}
//			}
//			LOG4CXX_INFO(logger, "End of input1 match " << diff);
//			diff = 0;
//			for (int j = 0; j < action1.size(); j ++) {
//				if (action1[j] != rb.actions[index][j]) {
//					LOG4CXX_INFO(logger, "mismatch " << j << ": " << action1[j] << ", " << rb.actions[index][j]);
//					diff ++;
//				}
//			}
//			LOG4CXX_INFO(logger, "End of action1 match " << diff);
//			LOG4CXX_INFO(logger, "Output equal: " << (outputTensor.equal(output1)));
//			LOG4CXX_INFO(logger, "update output: " << outputTensor);
//		}
		torch::Tensor target = std::get<0>(nextOutputTensor.max(-1)).unsqueeze(-1) * gamma;
		target = target * targetMask + rewardTensor;

		auto loss = lossComputer(lossInput, target);
		auto lossValue = loss.item<float>();
		optimizer.zero_grad();
		loss.backward();
		for (auto& param: model.parameters()) {
			param.clamp(-1, 1);
		}
		optimizer.step();
//		LOG4CXX_INFO(logger, "step " << i << " loss = " << lossValue);
	}

	updateNum ++;
	if (updateNum % UpdateExploreGap == 0) {
		policy.updateEpsilon(std::max<float>((float)policy.getEpsilon() - 0.1, 0.1f));
	}

	updateTarget();
}

//TODO: action tensor nograd
template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void DqnTarget<NetType, EnvType, PolicyType, OptimizerType>::train(const int batchSize, const int epochNum) {
	const int stepGap = rb.capacity;
	int step = 0;
	int epCount = 0;
	const int renderGap = 1;

	std::vector<float> statRewards(batchSize, 0);
	std::vector<float> statLens(batchSize, 0);

//	model.eval();
	std::vector<float> stateVec = env.reset();
	while (epCount < epochNum) {
		torch::Tensor inputTensor = torch::from_blob(stateVec.data(), inputShape).to(deviceType);
		torch::Tensor outputTensor = model.forward(inputTensor);

		std::vector<int64_t> actions = policy.getActions(outputTensor);
//		if (rb.getIndex() == 0) {
////			LOG4CXX_INFO(logger, "train step 1: " << outputTensor);
//			LOG4CXX_INFO(logger, "train step 1: " << actions);
//
//			input1.clear();
//			action1.clear();
//			for (int i = 0; i < stateVec.size(); i ++) {
//				input1.push_back(stateVec[i]);
//			}
//			for (int i = 0; i < actions.size(); i ++) {
//				actions[i] = i % 18;
//				action1.push_back(actions[i]);
//			}
//			output1 = torch::zeros(outputTensor.sizes()).to(deviceType);
//			{
//				torch::NoGradGuard gd;
//				output1.copy_(outputTensor);
//				LOG4CXX_INFO(logger, "train output: " << output1);
//			}
//		}
		bool render = false;
//		if (step % renderGap == 0) {
//			render = true;
//		}
		auto stepResult = env.step(actions, render);
		auto nextStateVec = std::get<0>(stepResult);
		auto rewardVec = std::get<1>(stepResult);
		auto doneVec = std::get<2>(stepResult);

		Stats::UpdateReward(statRewards, rewardVec);
		Stats::UpdateLen(statLens);

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
				LOG4CXX_INFO(logger, policy.getEpsilon() << stater);
			}
		}

		rb.put(stateVec, nextStateVec, rewardVec, actions, doneVec);
    	stateVec = nextStateVec;

		step ++;
//		LOG4CXX_INFO(logger, "step: " << step);
		if (step >= stepGap) {
			step = 0;

			update(batchSize);
//			model.eval();
//			test(batchSize, batchSize);
		}
	}
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void DqnTarget<NetType, EnvType, PolicyType, OptimizerType>::test(const int batchSize, const int epochNum) {
	model.eval();
	{
	torch::NoGradGuard gd;

	int epCount = 0;
	std::vector<float> statRewards(batchSize, 0);
	std::vector<float> statLens(batchSize, 0);

	std::vector<float> stateVec = testEnv.reset();
	while (epCount < epochNum) {
		torch::Tensor inputTensor = torch::from_blob(stateVec.data(), inputShape).to(deviceType);
		torch::Tensor outputTensor = model.forward(inputTensor);
		std::vector<int64_t> actions = policy.getTestActions(outputTensor);
//		std::vector<int64_t> actions = policy.getActions(outputTensor);
//		std::vector<int64_t> maxActions = policy.getTestActions(outputTensor);
//		LOG4CXX_DEBUG(logger, "test actions: " << actions);
//		LOG4CXX_INFO(logger, "actions: " << actions);
//		LOG4CXX_INFO(logger, "test as: " << maxActions);

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
	}
	model.train();
}
#endif /* INC_ALG_DQN_HPP_ */


