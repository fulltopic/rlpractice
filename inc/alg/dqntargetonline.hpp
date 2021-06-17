/*
 * dqntargetonline.hpp
 *
 *  Created on: Apr 14, 2021
 *      Author: zf
 */

#ifndef INC_ALG_DQNTARGETONLINE_HPP_
#define INC_ALG_DQNTARGETONLINE_HPP_


#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <iostream>
#include <vector>
#include <cstdint>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/replaybuffer.h"
#include "gymtest/utils/stats.h"
#include "dqnoption.h"

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
class DqnTargetOnline {
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

//	std::vector<float> input1;
//	std::vector<int64_t> action1;
//	torch::Tensor output1;

	uint32_t updateNum = 0;
	const uint32_t UpdateExploreGap = 1024;
	const int updateTargetGap; //TODO

	void update(std::vector<float>& inputVec, std::vector<float>& nextInputVec,
			std::vector<float>& rewardVec, std::vector<int64_t>& actionVec, std::vector<bool>& doneVec);
	void test(const int batchSize, const int epochNum);

	void updateTarget();
public:
	const float gamma;
	const int testEp = 16;

	DqnTargetOnline(NetType& iModel, NetType& tModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, DqnOption option);
	~DqnTargetOnline() = default;

	void train(const int batchSize, const int epochNum);
};

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
DqnTargetOnline<NetType, EnvType, PolicyType, OptimizerType>::DqnTargetOnline(NetType& iModel, NetType& tModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer,
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
	testStater("./test_stat.txt"),
	updateTargetGap(option.targetUpdate)
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
void DqnTargetOnline<NetType, EnvType, PolicyType, OptimizerType>::updateTarget() {
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
void DqnTargetOnline<NetType, EnvType, PolicyType, OptimizerType>::update(
		std::vector<float>& inputVec, std::vector<float>& nextInputVec,
		std::vector<float>& rewardVec, std::vector<int64_t>& actionVec, std::vector<bool>& doneVec) {
	LOG4CXX_DEBUG(logger, "---------------------> To update");
//	model.train();
//	model.eval();

//	LOG4CXX_INFO(logger, "action vec: " << rb.actions[index]);
	torch::Tensor stateTensor = torch::from_blob(inputVec.data(), inputShape).to(deviceType);
	LOG4CXX_DEBUG(logger, "created state tensor");
	torch::Tensor nextStateTensor = torch::from_blob(nextInputVec.data(), inputShape).to(deviceType);
	torch::Tensor rewardTensor = torch::from_blob(rewardVec.data(), {inputShape[0], 1}).to(deviceType);
	torch::Tensor actionTensor = torch::from_blob(actionVec.data(), {inputShape[0], 1}, longOpt).to(deviceType);
	LOG4CXX_DEBUG(logger, "created action tensor");
	auto targetMask = torch::ones({inputShape[0], 1}).to(deviceType);
	for (int j = 0; j < doneVec.size(); j ++) {
		    if (doneVec[j]) {
		    	targetMask[j][0] = 0;
		   }
	}

	torch::Tensor outputTensor = model.forward(stateTensor);
	torch::Tensor nextOutputTensor = targetModel.forward(nextStateTensor).detach();
	torch::Tensor lossInput = outputTensor.gather(-1, actionTensor);

	torch::Tensor target = std::get<0>(nextOutputTensor.max(-1)).unsqueeze(-1) * gamma;
	target = target * targetMask + rewardTensor;

	auto loss = lossComputer(lossInput, target);
	loss.clamp(-1, 1);
	auto lossValue = loss.item<float>();
	optimizer.zero_grad();
	loss.backward();
	for (auto& param: model.parameters()) {
		param.clamp(-1, 1);
	}
	optimizer.step();
//		LOG4CXX_INFO(logger, "step " << i << " loss = " << lossValue);
//	}

	updateNum ++;
	if (updateNum % UpdateExploreGap == 0) {
		policy.updateEpsilon(std::max<float>((float)policy.getEpsilon() - 0.1, 0.1f));
	}

	if ((updateNum % updateTargetGap) == 0) {
		updateTarget();
	}
}

//TODO: action tensor nograd
template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void DqnTargetOnline<NetType, EnvType, PolicyType, OptimizerType>::train(const int batchSize, const int epochNum) {
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

		update(stateVec, nextStateVec, rewardVec, actions, doneVec);


    	stateVec = nextStateVec;

		step ++;

	}
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void DqnTargetOnline<NetType, EnvType, PolicyType, OptimizerType>::test(const int batchSize, const int epochNum) {
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


