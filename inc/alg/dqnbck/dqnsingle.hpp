/*
 * dqnsingle.hpp
 *
 *  Created on: Apr 15, 2021
 *      Author: zf
 */

#ifndef INC_ALG_DQNSINGLE_HPP_
#define INC_ALG_DQNSINGLE_HPP_




#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <iostream>
#include <vector>
#include <cstdint>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/replaybuffer.h"
#include "gymtest/utils/nobatchrb.h"
#include "gymtest/utils/stats.h"
#include "dqnoption.h"

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
class DqnSingle {
private:
	NetType& model;
	NetType& targetModel;
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

	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("dqnsingle");
	torch::nn::SmoothL1Loss lossComputer = torch::nn::SmoothL1Loss();
	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);

//	std::vector<float> input1;
//	std::vector<int64_t> action1;
//	torch::Tensor output1;

	uint32_t updateNum = 0;
	uint32_t updateExploreGap = 1024;
	const int updateTargetGap; //TODO

	int64_t updateParamNum = 0;
	const int64_t UpParamTh = 10000;
	const int testUpdateNum = 100;
	int targetUpdated = 0;

	const DqnOption dqnOption;

	void update(const int batchSize);

	void updateTarget();
	bool toUpdateTarget();

	void test();

public:
	const float gamma;
	const int testEp = 4;

	DqnSingle(NetType& iModel, NetType& tModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, DqnOption option);
	~DqnSingle() = default;

	void train(const int batchSize, const int epochNum);

	void save();
	void load();
};

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
DqnSingle<NetType, EnvType, PolicyType, OptimizerType>::DqnSingle(NetType& iModel, NetType& tModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer,
		const DqnOption iOption):
	model(iModel),
	targetModel(tModel),
	env(iEnv),
	testEnv(tEnv),
	policy(iPolicy),
	optimizer(iOptimizer),
	dqnOption(iOption),
	deviceType(iOption.deviceType),
	inputShape(iOption.inputShape),
	rb(iOption.rbCap),
	gamma(iOption.gamma),
	stater(iOption.statPath),
	testStater(iOption.teststatPath),
	updateTargetGap(iOption.targetUpdate)
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
void DqnSingle<NetType, EnvType, PolicyType, OptimizerType>::test() {
	if (!dqnOption.toTest) {
		return;
	}
	int epCount = 0;
	const int batchSize = 1;

	std::vector<float> statRewards(batchSize, 0);
	std::vector<float> statLens(batchSize, 0);

//	model.eval();
	std::vector<float> stateVec = testEnv.reset();
	while (epCount < testEp) {
		torch::Tensor inputTensor = torch::from_blob(stateVec.data(), inputShape).to(deviceType);
		torch::Tensor outputTensor = model.forward(inputTensor);

		std::vector<int64_t> actions = policy.getTestActions(outputTensor);

		bool render = false;
//		if (step % renderGap == 0) {
//			render = true;
//		}
		auto stepResult = testEnv.step(actions, render);
		auto nextStateVec = std::get<0>(stepResult);
		auto rewardVec = std::get<1>(stepResult);
		auto doneVec = std::get<2>(stepResult);

		Stats::UpdateReward(statRewards, rewardVec);
		Stats::UpdateLen(statLens);

		if (doneVec[0]) {
			LOG4CXX_DEBUG(logger, "ep " << epCount << "done");
//			auto resetResult = testEnv.reset();
			//udpate nextstatevec, target mask
//			std::copy(resetResult.begin(), resetResult.end(), nextStateVec.begin());
			epCount ++;

			testStater.update(statLens[0], statRewards[0]);
			statLens[0] = 0;
			statRewards[0] = 0;
			LOG4CXX_INFO(logger, "test " << testStater);
		}

    	stateVec = nextStateVec;
	}
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
bool DqnSingle<NetType, EnvType, PolicyType, OptimizerType>::toUpdateTarget() {
	int64_t updatedNum = 0;

	auto paramDict = model.named_parameters();
	auto buffDict = model.named_buffers();
	auto targetParamDict = targetModel.named_parameters();
	auto targetBuffDict = targetModel.named_buffers();

	for(const auto& item: paramDict) {
		const auto& key = item.key();
		const auto& param = item.value();
		const auto& targetParam = targetParamDict[key];
		torch::Tensor diff = (targetParam - param).count_nonzero();
		updatedNum += diff.item<int64_t>();
	}

	for (const auto& item: buffDict) {
		const auto& key = item.key();
		const auto& buff = item.value();
		const auto& targetBuff = targetBuffDict[key];
		torch::Tensor diff = targetBuff - buff;
		updatedNum += diff.count_nonzero().item<int64_t>();
	}
//	LOG4CXX_INFO(logger, "updated: " << updatedNum);

	updateParamNum += updatedNum;
	if (updateParamNum >= UpParamTh) {
		updateParamNum = 0;
		return true;
	} else {
		return false;
	}
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void DqnSingle<NetType, EnvType, PolicyType, OptimizerType>::updateTarget() {
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

	targetUpdated ++;

	if (targetUpdated == testUpdateNum) {
		targetUpdated = 0;
		test();
	}
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void DqnSingle<NetType, EnvType, PolicyType, OptimizerType>::update(const int batchSize) {
	if (rb.getCount() < batchSize) {
		return;
	}
	LOG4CXX_DEBUG(logger, "---------------------> To update");
	std::vector<int64_t> stateShape(inputShape.size(), 0);
	stateShape[0] = 1;
	for (int i = 1; i < stateShape.size(); i ++) {
		stateShape[i] = inputShape[i];
	}
//	model.train();


	std::vector<torch::Tensor> subStates;
	std::vector<torch::Tensor> subNextStates;
	std::vector<float> subRewards;
	std::vector<long> subActions;
	std::vector<bool> subDones;

	for (int j = 0; j < batchSize; j ++) {
		int index = rb.randSelect();

		subStates.push_back(torch::from_blob(rb.states[index].data(), stateShape).to(deviceType));
		subNextStates.push_back(torch::from_blob(rb.nextStates[index].data(), stateShape).to(deviceType));
		subRewards.push_back(rb.rewards[index]);
		subActions.push_back(rb.actions[index]);
		subDones.push_back(rb.dones[index]);
	}
	torch::Tensor stateTensor = torch::cat(subStates, 0).div(dqnOption.inputScale);
	torch::Tensor nextStateTensor = torch::cat(subNextStates, 0).div(dqnOption.inputScale);
	torch::Tensor rewardTensor = torch::from_blob(subRewards.data(), {batchSize, 1}).div(dqnOption.rewardScale).clamp(dqnOption.rewardMin, dqnOption.rewardMax).to(deviceType);
	torch::Tensor actionTensor = torch::from_blob(subActions.data(), {batchSize, 1}, longOpt).to(deviceType);

//	torch::Tensor diff = torch::count_nonzero(nextStateTensor - stateTensor);
//	LOG4CXX_INFO(logger, "diff: " << diff);

	torch::Tensor targetMask = torch::ones({batchSize, 1}).to(deviceType);
	for (int j = 0; j < batchSize; j ++) {
		if (subDones[j]) {
			targetMask[j][0] = 0;
		}
	}
//	rewardTensor = rewardTensor.clamp(-1, 1); //after test20

	torch::Tensor outputTensor = model.forward(stateTensor);
	torch::Tensor nextOutputTensor = targetModel.forward(nextStateTensor).detach();
	torch::Tensor lossInput = outputTensor.gather(-1, actionTensor);
	torch::Tensor target = std::get<0>(nextOutputTensor.max(-1)).unsqueeze(-1) * gamma;
//	rewardTensor = rewardTensor.clamp(-1, 1); //after test17
	target = target * targetMask + rewardTensor;

	auto loss = lossComputer(lossInput, target);
//	LOG4CXX_INFO(logger, "loss" << updateNum << ": " << loss.item<float>());
	//TODO: impose a clamp on reward
	//annotated since test19
//	loss = loss.clamp(-1, 1); //after test17
//	auto lossValue = loss.item<float>();
	optimizer.zero_grad();
	loss.backward();
//	for (auto& param: model.parameters()) {
//		param.clamp(-1, 1);
//	}
	torch::nn::utils::clip_grad_value_(model.parameters(), 1);
//	for (auto& param: model.parameters()) {
//		param.grad().clamp(-1, 1);
//	}
	optimizer.step();


	updateNum ++;
//	if (updateNum % updateExploreGap == 0) {
//		policy.updateEpsilon(std::max<float>((float)policy.getEpsilon() - 0.1, 0.1f));
//	}

//	if (toUpdateTarget()) {
//		updateTarget();
//	}
	if ((updateNum % updateTargetGap) == 0) {
		updateTarget();
	}
}

//TODO: action tensor nograd
template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void DqnSingle<NetType, EnvType, PolicyType, OptimizerType>::train(const int batchSize, const int epochNum) {
	load();

//	updateExploreGap = (epochNum / 2) / 9;
//	updateExploreGap = std::max<float>(dqnOption.exploreEp / dqnOption.explorePhase, 1);
//	const int stepGap = rb.capacity;
	int step = 0;
	int epCount = 0;
	const int clientNum = 1;
//	const int renderGap = 1;

	std::vector<float> statRewards(clientNum, 0);
	std::vector<float> statLens(clientNum, 0);

//	model.eval();
	std::vector<float> stateVec = env.reset();
	while (epCount < epochNum) {
		torch::Tensor inputTensor = torch::from_blob(stateVec.data(), inputShape).to(deviceType).div(dqnOption.inputScale);
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

		if (doneVec[0]) {
			LOG4CXX_DEBUG(logger, "ep " << epCount << "done");
//			auto resetResult = env.reset();
			//udpate nextstatevec, target mask
//			std::copy(resetResult.begin(), resetResult.end(), nextStateVec.begin());
			epCount ++;

			stater.update(statLens[0], statRewards[0]);
			statLens[0] = 0;
			statRewards[0] = 0;
			LOG4CXX_INFO(logger, policy.getEpsilon() << stater);

//			if (epCount % updateExploreGap == 0) {
//				policy.updateEpsilon(std::max<float>((float)policy.getEpsilon() - dqnOption.exploreDecay, 0.1f));
//			}

		}
		rb.put(stateVec, nextStateVec, rewardVec, actions, doneVec, 1);
    	stateVec = nextStateVec;

    	if (step < dqnOption.exploreStep) {
    		float exploreRate = dqnOption.exploreBegin - (dqnOption.exploreBegin - dqnOption.exploreEnd) * (step / dqnOption.exploreStep);
    		policy.updateEpsilon(exploreRate);
    	}
    	if (step >= dqnOption.startStep) {
    		update(batchSize);
    	}


		step ++;
	}

	save();
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void DqnSingle<NetType, EnvType, PolicyType, OptimizerType>::save() {
	if (!dqnOption.saveModel) {
		return;
	}

	std::string modelPath = dqnOption.savePathPrefix + "_model.pt";
	torch::serialize::OutputArchive outputArchive;
	model.save(outputArchive);
	outputArchive.save_to(modelPath);
	LOG4CXX_INFO(logger, "Save model into " << modelPath);

	std::string optPath = dqnOption.savePathPrefix + "_optimizer.pt";
	torch::serialize::OutputArchive optimizerArchive;
	optimizer.save(optimizerArchive);
	optimizerArchive.save_to(optPath);
	LOG4CXX_INFO(logger, "Save optimizer into " << optPath);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void DqnSingle<NetType, EnvType, PolicyType, OptimizerType>::load() {
	if (!dqnOption.loadModel) {
		return;
	}

	std::string modelPath = dqnOption.loadPathPrefix + "_model.pt";
	torch::serialize::InputArchive inChive;
	inChive.load_from(modelPath);
	model.load(inChive);
	LOG4CXX_INFO(logger, "Load model from " << modelPath);

	updateTarget();

	if (dqnOption.loadOptimizer) {
		std::string optPath = dqnOption.loadPathPrefix + "_optimizer.pt";
		torch::serialize::InputArchive opInChive;
		opInChive.load_from(optPath);
		optimizer.load(opInChive);
		LOG4CXX_INFO(logger, "Load optimizer from " << optPath);
	}

}

#endif /* INC_ALG_DQNSINGLE_HPP_ */
