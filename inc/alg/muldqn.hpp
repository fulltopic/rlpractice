/*
 * muldqn.hpp
 *
 *  Created on: Aug 28, 2021
 *      Author: zf
 */

#ifndef INC_ALG_MULDQN_HPP_
#define INC_ALG_MULDQN_HPP_


#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/stats.h"
#include "gymtest/utils/lossstats.h"
#include "dqnoption.h"

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
class MultiDqn {
private:
	NetType& bModel;
	NetType& tModel;
	EnvType& env;
	EnvType& testEnv;
	PolicyType& policy;
	OptimizerType& optimizer;
	const torch::Device deviceType;
	const at::IntArrayRef inputShape;

	const DqnOption dqnOption;

	uint32_t updateNum = 0;

//	const int actionNum;
//	std::vector<int64_t> indice;

	const torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("dqn");

	Stats stater;
	Stats testStater;
	LossStats lossStater;


	class ReplayBuffer {
	private:
		int curIndex = 0;
		int curSize = 0;
		const int cap;

		const torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
		log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("dqn");

	public:
		ReplayBuffer (const int iCap, const at::IntArrayRef& inputShape);
		~ReplayBuffer() = default;
		ReplayBuffer(const ReplayBuffer&) = delete;

		torch::Tensor states;
		torch::Tensor actions;
		torch::Tensor rewards;
		torch::Tensor donesMask;

		//Store states and rewards after normalization
		void add(torch::Tensor state, torch::Tensor nextState, int action, float reward, float done);
		torch::Tensor getSampleIndex(int batchSize);
	};

	ReplayBuffer buffer; //buffer has to be defined after dqnOption so ReplayBuffer can get all parameters of dqnOption.

	void updateModel(bool force = false);
	void updateStep(const float epochNum);

	void load();
	void save();
public:
	MultiDqn(NetType& iModel, NetType& iTModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, DqnOption iOption);
	~MultiDqn() = default;
	MultiDqn(const Dqn&) = delete;

	void train(const int epochNum);
	void test(const int epochNum, bool render = false);
};



template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
MultiDqn<NetType, EnvType, PolicyType, OptimizerType>::ReplayBuffer::ReplayBuffer(const int iCap, const at::IntArrayRef& inputShape): cap(iCap) {
	std::vector<int64_t> stateInputShape;
	stateInputShape.push_back(cap);
	//input state shape = {1, 4, 84, 84};
	for (int i = 1; i < inputShape.size(); i ++) {
		stateInputShape.push_back(inputShape[i]);
	}
	at::IntArrayRef outputShape{ReplayBuffer::cap, 1};

	states = torch::zeros(stateInputShape);
	actions = torch::zeros(outputShape, longOpt);
	rewards = torch::zeros(outputShape);
	donesMask = torch::zeros(outputShape);

	LOG4CXX_DEBUG(logger, "Replay buffer ready");
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void MultiDqn<NetType, EnvType, PolicyType, OptimizerType>::ReplayBuffer::add(
		torch::Tensor state, torch::Tensor nextState, int action, float reward, float done) {
	{
		//For log
		bool isSame = states[curIndex].equal(state);
		LOG4CXX_DEBUG(logger, curIndex << ": The state same? " << isSame);
		if (!isSame) {
			LOG4CXX_DEBUG(logger, "curState: " << states[curIndex]);
			LOG4CXX_DEBUG(logger, "inputState: " << state);
			LOG4CXX_DEBUG(logger, "nextState: " << nextState);
		}
	}

	int nextIndex = (curIndex + 1) % cap;

	states[curIndex].copy_(state.squeeze());
	states[nextIndex].copy_(nextState.squeeze()); //TODO: Optimize
	actions[curIndex][0] = action;
	rewards[curIndex][0] = reward;
	donesMask[curIndex][0] = done;
	LOG4CXX_DEBUG(logger, "states after copy: " << states[curIndex]);

	curIndex = nextIndex;
	if (curSize < cap) {
		curSize ++;
	}
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
torch::Tensor MultiDqn<NetType, EnvType, PolicyType, OptimizerType>::ReplayBuffer::getSampleIndex(int batchSize) {
	torch::Tensor indices = torch::randint(0, curSize, {batchSize}, longOpt);

	return indices;
}


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
MultiDqn<NetType, EnvType, PolicyType, OptimizerType>::MultiDqn(NetType& iModel, NetType& iTModel,
		EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer,
		DqnOption iOption):
	bModel(iModel),
	tModel(iTModel),
	env(iEnv),
	testEnv(tEnv),
	policy(iPolicy),
	optimizer(iOptimizer),
	dqnOption(iOption),
	deviceType(iOption.deviceType),
	inputShape(iOption.inputShape),
	buffer(iOption.rbCap, iOption.inputShape),
	stater(iOption.statPathPrefix + "_stat.txt", iOption.statCap),
	testStater(iOption.statPathPrefix + "_test.txt", iOption.statCap),
	lossStater(iOption.statPathPrefix + "_loss.txt")
{

}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void MultiDqn<NetType, EnvType, PolicyType, OptimizerType>::train(const int epochNum) {
	load();
	updateModel(true); //model assignment
	tModel.eval();

	std::vector<float> stateVec = env.reset();
//	std::vector<float> nextStateVec;

	//only one env
	std::vector<float> statRewards(dqnOption.envNum, 0);
	std::vector<float> statLens(dqnOption.envNum, 0);

	while (updateNum < epochNum) {
		updateNum ++;
		//Run step
		torch::Tensor cpuinputTensor = torch::from_blob(stateVec.data(), inputShape).div(dqnOption.inputScale);
		torch::Tensor inputTensor = cpuinputTensor.to(deviceType);

		torch::Tensor outputTensor = bModel.forward(inputTensor); //TODO: bModel or tModel?
		std::vector<int64_t> actions = policy.getActions(outputTensor);

		auto stepResult = env.step(actions);
		auto nextInputVec = std::get<0>(stepResult);
		auto rewardVec = std::get<1>(stepResult);
		auto doneVec = std::get<2>(stepResult);

		Stats::UpdateReward(statRewards, rewardVec);
		Stats::UpdateLen(statLens);
		float doneMask = 1;
		if (doneVec[0]) {
			doneMask = 0;

			stater.update(statLens[0], statRewards[0]);
			statRewards[0] = 0;
			statLens[0] = 0;
			LOG4CXX_INFO(logger, "" << policy.getEpsilon() << "--" << updateNum << stater);
		}

		torch::Tensor nextInputTensor = torch::from_blob(nextInputVec.data(), inputShape).div(dqnOption.inputScale);
		float reward = std::max(std::min((rewardVec[0] / dqnOption.rewardScale), dqnOption.rewardMax), dqnOption.rewardMin);
		buffer.add(cpuinputTensor, nextInputTensor, actions[0], rewardVec[0], doneMask);

		//Update
		stateVec = nextInputVec;
		updateStep(epochNum);
		//Learning
		if (updateNum < dqnOption.startStep) {
			continue;
		}

		torch::Tensor sampleIndice = buffer.getSampleIndex(dqnOption.batchSize);
		torch::Tensor curStateTensor = buffer.states.index_select(0, sampleIndice).to(deviceType);
		torch::Tensor actionTensor = buffer.actions.index_select(0, sampleIndice).to(deviceType);
		torch::Tensor rewardTensor = buffer.rewards.index_select(0, sampleIndice).to(deviceType);
		torch::Tensor doneMaskTensor = buffer.donesMask.index_select(0, sampleIndice).to(deviceType);
		LOG4CXX_DEBUG(logger, "sampleIndex before: " << sampleIndice);
		sampleIndice = (sampleIndice + 1) % dqnOption.rbCap;
		torch::Tensor nextStateTensor = buffer.states.index_select(0, sampleIndice).to(deviceType);
		LOG4CXX_DEBUG(logger, "sampleIndice after: " << sampleIndice);
		LOG4CXX_DEBUG(logger, "nextStateTensor: " << nextStateTensor);

		torch::Tensor targetQ;
		LOG4CXX_DEBUG(logger, "targetQ before " << targetQ);
		{
			torch::NoGradGuard guard; //replaced by tModel.eval()?
			torch::Tensor nextOutput = tModel.forward(nextStateTensor).detach();
			LOG4CXX_DEBUG(logger, "nextOutput: " << nextOutput);
			auto maxOutput = nextOutput.max(-1);
			torch::Tensor nextQ = std::get<0>(maxOutput); //TODO: pay attention to shape
			nextQ = nextQ.unsqueeze(1);
			LOG4CXX_DEBUG(logger, "nextQ: " << nextQ);
			LOG4CXX_DEBUG(logger, "rewardTensor: " << rewardTensor);
			LOG4CXX_DEBUG(logger, "doneMaskTensor: " << doneMaskTensor);
			targetQ = rewardTensor + dqnOption.gamma * nextQ * doneMaskTensor;
			LOG4CXX_DEBUG(logger, "targetQ: " << targetQ);
		}

		torch::Tensor curOutput = bModel.forward(curStateTensor);
		LOG4CXX_DEBUG(logger, "curOutput: " << curOutput);
		torch::Tensor curQ = curOutput.gather(-1, actionTensor); //TODO: shape of actionTensor and curQ
		LOG4CXX_DEBUG(logger, "curQ: " << curQ);

//		auto loss = torch::nn::SmoothL1Loss(curQ, targetQ);
		//TODO: Try mse.
		auto loss = torch::nn::functional::smooth_l1_loss(curQ, targetQ);

		if ((updateNum % dqnOption.logInterval) == 0) {
			float lossValue = loss.item<float>();
			auto curStat = stater.getCurState();
			lossStater.update({(float)updateNum, lossValue, curStat[0], curStat[1]});
		}

		optimizer.zero_grad();
		loss.backward();
		torch::nn::utils::clip_grad_norm_(bModel.parameters(), dqnOption.maxGradNormClip);
		optimizer.step();
	}

	save();
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void MultiDqn<NetType, EnvType, PolicyType, OptimizerType>::updateModel(bool force) {
	if (!force) {
		if ((updateNum % dqnOption.targetUpdateStep) != 0) {
			return;
		}
	}

	torch::NoGradGuard guard;

	auto paramDict = bModel.named_parameters();
	auto buffDict = bModel.named_buffers();
	auto targetParamDict = tModel.named_parameters();
	auto targetBuffDict = tModel.named_buffers();

	for (const auto& item: paramDict) {
		const auto& key = item.key();
		const auto param = item.value();
		auto& targetParam = targetParamDict[key];

		targetParam.mul_(1 - dqnOption.tau);
		targetParam.add_(param, dqnOption.tau);
	}

	for (const auto& item: buffDict) {
		const auto& key = item.key();
		const auto& buff = item.value();
		auto& targetBuff = targetBuffDict[key];

		targetBuff.mul(1 - dqnOption.tau);
		targetBuff.add_(buff, dqnOption.tau);
	}
	LOG4CXX_INFO(logger, "target network synched");
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void MultiDqn<NetType, EnvType, PolicyType, OptimizerType>::updateStep(const float epochNum) {
	updateModel(false);

	if (updateNum > (dqnOption.explorePart * epochNum)) {
		return;
	}
	float newEpsilon = (dqnOption.exploreBegin - dqnOption.exploreEnd) * (epochNum * dqnOption.explorePart - updateNum) / (epochNum * dqnOption.explorePart) + dqnOption.exploreEnd;
	policy.updateEpsilon(newEpsilon);
}
//TODO: update, train, test, syncModel

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void MultiDqn<NetType, EnvType, PolicyType, OptimizerType>::test(const int epochNum, bool render) {
	load();

	std::vector<float> statRewards(dqnOption.envNum, 0);
	std::vector<float> statLens(dqnOption.envNum, 0);

	std::vector<float> stateVec = env.reset();

	while (updateNum < epochNum) {
		torch::Tensor inputTensor = torch::from_blob(stateVec.data(), inputShape).div(dqnOption.inputScale).to(deviceType);
		torch::Tensor outputTensor = bModel.forward(inputTensor);
		std::vector<int64_t> actions = policy.getTestActions(outputTensor);

		auto stepResult = env.step(actions, render);
		auto nextInputVec = std::get<0>(stepResult);
		auto rewardVec = std::get<1>(stepResult);
		auto doneVec = std::get<2>(stepResult);

		Stats::UpdateReward(statRewards, rewardVec);
		Stats::UpdateLen(statLens);
		float doneMask = 1;
		if (doneVec[0]) {
			doneMask = 0;

			testStater.update(statLens[0], statRewards[0]);
			statRewards[0] = 0;
			statLens[0] = 0;
			LOG4CXX_INFO(logger, testStater);
		}

		stateVec = nextInputVec;
		updateNum ++;
	}
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void MultiDqn<NetType, EnvType, PolicyType, OptimizerType>::save() {
	if (!dqnOption.saveModel) {
		return;
	}

	std::string modelPath = dqnOption.savePathPrefix + "_model.pt";
	torch::serialize::OutputArchive outputArchive;
	bModel.save(outputArchive);
	outputArchive.save_to(modelPath);
	LOG4CXX_INFO(logger, "Save model into " << modelPath);

	std::string optPath = dqnOption.savePathPrefix + "_optimizer.pt";
	torch::serialize::OutputArchive optimizerArchive;
	optimizer.save(optimizerArchive);
	optimizerArchive.save_to(optPath);
	LOG4CXX_INFO(logger, "Save optimizer into " << optPath);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void MultiDqn<NetType, EnvType, PolicyType, OptimizerType>::load() {
	if (!dqnOption.loadModel) {
		return;
	}

	std::string modelPath = dqnOption.loadPathPrefix + "_model.pt";
	torch::serialize::InputArchive inChive;
	inChive.load_from(modelPath);
	bModel.load(inChive);
	LOG4CXX_INFO(logger, "Load model from " << modelPath);

//	updateTarget();

	if (dqnOption.loadOptimizer) {
		std::string optPath = dqnOption.loadPathPrefix + "_optimizer.pt";
		torch::serialize::InputArchive opInChive;
		opInChive.load_from(optPath);
		optimizer.load(opInChive);
		LOG4CXX_INFO(logger, "Load optimizer from " << optPath);
	}

}


#endif /* INC_ALG_MULDQN_HPP_ */
