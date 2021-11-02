/*
 * catdqn.hpp
 *
 *  Created on: Oct 21, 2021
 *      Author: zf
 */

#ifndef INC_ALG_CATDQN_HPP_
#define INC_ALG_CATDQN_HPP_



#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>
#include <tensorboard_logger.h>

#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/stats.h"
#include "gymtest/utils/lossstats.h"
#include "dqnoption.h"

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
class CategoricalDqn {
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
	uint32_t totalTestEp = 0;

	float maxTestReward = -1000;

	torch::Tensor valueItems;
	torch::Tensor offset;

	float deltaZ = 0;

	//log
	const torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("dqn");
	TensorBoardLogger tLogger;

	//test
	std::vector<float> statRewards = std::vector<float>(dqnOption.testBatch, 0);
	std::vector<float> statLens = std::vector<float>(dqnOption.testBatch, 0);
	std::vector<float> statEpRewards = std::vector<float>(dqnOption.testBatch, 0);
	std::vector<float> statEpLens = std::vector<float>(dqnOption.testBatch, 0);
	std::vector<int> livePerEp = std::vector<int>(dqnOption.testBatch, 0);

	//print out log
	Stats stater;
	Stats testStater;
	LossStats lossStater;

	class ReplayBuffer {
	private:
		int curIndex = 0;
		int curSize = 0;
		const int cap;

		const torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
		const torch::TensorOptions byteOpt = torch::TensorOptions().dtype(torch::kByte);
		const torch::TensorOptions charOpt = torch::TensorOptions().dtype(torch::kChar);

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
	void saveTModel(float reward);
public:
	CategoricalDqn(NetType& iModel, NetType& iTModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, DqnOption iOption);
	~CategoricalDqn() = default;
	CategoricalDqn(const CategoricalDqn&) = delete;

	void train(const int epochNum);
	void test(const int epochNum, bool render = false, bool toLoad = true);
};



template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
CategoricalDqn<NetType, EnvType, PolicyType, OptimizerType>::ReplayBuffer::ReplayBuffer(const int iCap, const at::IntArrayRef& inputShape): cap(iCap) {
	std::vector<int64_t> stateInputShape;
	stateInputShape.push_back(cap);
	//input state shape = {1, 4, 84, 84};
	for (int i = 1; i < inputShape.size(); i ++) {
		stateInputShape.push_back(inputShape[i]);
	}
	at::IntArrayRef outputShape{ReplayBuffer::cap, 1};

	states = torch::zeros(stateInputShape, byteOpt);
//	states = torch::zeros(stateInputShape);
	actions = torch::zeros(outputShape, byteOpt);
	rewards = torch::zeros(outputShape);
	donesMask = torch::zeros(outputShape, byteOpt);

	LOG4CXX_DEBUG(logger, "Replay buffer ready");
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void CategoricalDqn<NetType, EnvType, PolicyType, OptimizerType>::ReplayBuffer::add(
		torch::Tensor state, torch::Tensor nextState, int action, float reward, float done) {
		int nextIndex = (curIndex + 1) % cap;

		torch::Tensor inputState = state.to(torch::kByte);
		torch::Tensor inputNextState = nextState.to(torch::kByte);
//		auto inputState = state;
//		auto inputNextState = nextState;


		states[curIndex].copy_(inputState.squeeze());
		states[nextIndex].copy_(inputNextState.squeeze()); //TODO: Optimize
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
torch::Tensor CategoricalDqn<NetType, EnvType, PolicyType, OptimizerType>::ReplayBuffer::getSampleIndex(int batchSize) {
	torch::Tensor indices = torch::randint(0, curSize, {batchSize}, longOpt);

	return indices;
}


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
CategoricalDqn<NetType, EnvType, PolicyType, OptimizerType>::CategoricalDqn(NetType& iModel, NetType& iTModel,
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
	testStater(iOption.statPathPrefix + "_test.txt", iOption.testEp),
	lossStater(iOption.statPathPrefix + "_loss.txt"),
	tLogger(iOption.tensorboardLogPath.c_str())
{
	maxTestReward = iOption.saveThreshold;

	valueItems = torch::linspace(dqnOption.vMin, dqnOption.vMax, dqnOption.atomNum).to(dqnOption.deviceType);
	offset = (torch::linspace(0, dqnOption.batchSize - 1, dqnOption.batchSize) * dqnOption.atomNum).unsqueeze(-1).to(torch::kLong).to(dqnOption.deviceType); //{batchSize * atomNum, 1}
	deltaZ = (dqnOption.vMax - dqnOption.vMin) / ((float)(dqnOption.atomNum - 1));
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void CategoricalDqn<NetType, EnvType, PolicyType, OptimizerType>::train(const int epochNum) {
	load();
	updateModel(true); //model assignment
	tModel.eval();

	std::vector<float> stateVec = env.reset();

	//only one env
	std::vector<float> statRewards(dqnOption.envNum, 0);
	std::vector<float> statLens(dqnOption.envNum, 0);

	while (updateNum < epochNum) {
		for (int k = 0; k < dqnOption.envStep; k ++) {
			updateNum ++;
			//Run step
			torch::Tensor cpuinputTensor = torch::from_blob(stateVec.data(), inputShape);
			torch::Tensor inputTensor = cpuinputTensor.to(deviceType).div(dqnOption.inputScale);

			torch::Tensor outputTensor = bModel.forward(inputTensor);
			outputTensor = outputTensor.view({1, dqnOption.outputNum, dqnOption.atomNum});
			outputTensor = torch::softmax(outputTensor, -1).squeeze(0);
//			LOG4CXX_INFO(logger, "predict dist: " << outputTensor);
			outputTensor = outputTensor * valueItems; //ValueItems expanded properly
			outputTensor = outputTensor.sum(-1, false);
//			LOG4CXX_INFO(logger, "input state: " << inputTensor);
//			LOG4CXX_INFO(logger, "predict values: " << outputTensor);
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

				tLogger.add_scalar("train/reward", updateNum, statRewards[0]);
				tLogger.add_scalar("train/len", updateNum, statLens[0]);
				stater.update(statLens[0], statRewards[0]);
				statRewards[0] = 0;
				statLens[0] = 0;
				LOG4CXX_INFO(logger, "" << policy.getEpsilon() << "--" << updateNum << stater);
			}

			torch::Tensor nextInputTensor = torch::from_blob(nextInputVec.data(), inputShape); //.div(dqnOption.inputScale);
			float reward = rewardVec[0]; //Would be clipped in atom operation
			buffer.add(cpuinputTensor, nextInputTensor, actions[0], rewardVec[0], doneMask);

			stateVec = nextInputVec;
			updateStep(epochNum);
		} //End of envStep

		//Learning
		if (updateNum < dqnOption.startStep) {
			continue;
		}

		float updateLoss = 0;
		float updateQs = 0;
		for (int k = 0; k < dqnOption.epochPerUpdate; k ++) {
			//sample
			torch::Tensor sampleIndice = buffer.getSampleIndex(dqnOption.batchSize);
			torch::Tensor curStateTensor = buffer.states.index_select(0, sampleIndice).to(deviceType);
			torch::Tensor actionTensor = buffer.actions.index_select(0, sampleIndice).to(deviceType);
			torch::Tensor rewardTensor = buffer.rewards.index_select(0, sampleIndice).to(deviceType); //{batchSize, 1}
			torch::Tensor doneMaskTensor = buffer.donesMask.index_select(0, sampleIndice).to(deviceType);
			LOG4CXX_DEBUG(logger, "sampleIndex before: " << sampleIndice);
			auto nextSampleIndice = (sampleIndice + 1) % dqnOption.rbCap;
			torch::Tensor nextStateTensor = buffer.states.index_select(0, nextSampleIndice).to(deviceType);

			curStateTensor = curStateTensor.to(torch::kFloat).div(dqnOption.inputScale);
			nextStateTensor = nextStateTensor.to(torch::kFloat).div(dqnOption.inputScale);
			rewardTensor = rewardTensor.to(torch::kFloat);
			doneMaskTensor = doneMaskTensor.to(torch::kFloat);
			actionTensor = actionTensor.to(torch::kLong);

			LOG4CXX_DEBUG(logger, "sampleIndice after: " << sampleIndice);
			LOG4CXX_DEBUG(logger, "nextStateTensor: " << nextStateTensor);

			//Calculate target
			torch::Tensor targetDist;
			LOG4CXX_DEBUG(logger, "targetQ before " << targetDist);
			{
				torch::NoGradGuard guard;

				torch::Tensor nextOutput = tModel.forward(nextStateTensor).detach();
				nextOutput = nextOutput.view({dqnOption.batchSize, dqnOption.outputNum, dqnOption.atomNum});
				torch::Tensor nextProbs = torch::softmax(nextOutput, -1); //{batch, action, atom}

				torch::Tensor nextQs = nextProbs * valueItems; // valueItems expands
				nextQs = nextQs.sum(-1, false); //{batchSize, actionNum}
				auto nextMaxOutput = nextQs.max(-1, true); //{batchSize, actionNum}
				torch::Tensor nextMaxQs = std::get<0>(nextMaxOutput); //{batchSize, 1}
				torch::Tensor nextMaxActions = std::get<1>(nextMaxOutput); //{batchSize, 1}
				nextMaxActions = nextMaxActions.unsqueeze(1).expand({dqnOption.batchSize, 1, dqnOption.atomNum});
				torch::Tensor nextDist = nextProbs.gather(1, nextMaxActions).squeeze(1); //{batchSize, atomNum}

				torch::Tensor shiftValues = rewardTensor + dqnOption.gamma * doneMaskTensor * valueItems; //{batchSize, atomNum}
				shiftValues = shiftValues.clamp(dqnOption.vMin, dqnOption.vMax);
				torch::Tensor shiftIndex = (shiftValues - dqnOption.vMin) / deltaZ; //{batchSize, atomNum}
				torch::Tensor l = shiftIndex.floor();
				torch::Tensor u = shiftIndex.ceil();
				torch::Tensor lIndice = l.to(torch::kLong);
				torch::Tensor uIndice = u.to(torch::kLong);
				torch::Tensor lDelta = u - shiftIndex;
				torch::Tensor uDelta = shiftIndex - l;
				torch::Tensor eqIndice = lIndice.eq(uIndice).to(torch::kFloat); //bool to float
				lDelta.add_(eqIndice); //shiftIndex % deltaZ == 0

				//Prepare for index_add
				torch::Tensor lDist = (nextDist * lDelta).view({dqnOption.batchSize * dqnOption.atomNum});  //{batchSize, atomNum}
				torch::Tensor uDist = (nextDist * uDelta).view({dqnOption.batchSize * dqnOption.atomNum});

				targetDist = torch::zeros({dqnOption.batchSize * dqnOption.atomNum}).to(dqnOption.deviceType);

				lIndice = (lIndice + offset).view({dqnOption.batchSize * dqnOption.atomNum}); //offset expanded
				uIndice = (uIndice + offset).view({dqnOption.batchSize * dqnOption.atomNum});
				targetDist.index_add_(0, lIndice, lDist);
				targetDist.index_add_(0, uIndice, uDist);

				targetDist = targetDist.view({dqnOption.batchSize, dqnOption.atomNum});
//				LOG4CXX_INFO(logger, "targetDist: " << targetDist);
//				LOG4CXX_INFO(logger, "new dist: " << targetDist.sum(-1));
			}

			//Calculate current Q
			torch::Tensor curOutput = bModel.forward(curStateTensor);
			curOutput = curOutput.view({dqnOption.batchSize, dqnOption.outputNum, dqnOption.atomNum});
			actionTensor = actionTensor.unsqueeze(1).expand({dqnOption.batchSize, 1, dqnOption.atomNum});
			torch::Tensor curDist = curOutput.gather(1, actionTensor).squeeze(1); //{batchSize, atomNum}
			torch::Tensor curLogDist = torch::log_softmax(curDist, -1);
			LOG4CXX_DEBUG(logger, "curOutput: " << curOutput);
//			LOG4CXX_INFO(logger, "curLogDist: " << curLogDist);

			//Update
			auto loss = -(targetDist * curLogDist).sum(-1).mean();

			optimizer.zero_grad();
			loss.backward();
			torch::nn::utils::clip_grad_norm_(bModel.parameters(), dqnOption.maxGradNormClip);
			optimizer.step();

			updateLoss += loss.item<float>();
			updateQs += (curDist.softmax(-1) * valueItems).sum(-1, false).mean().item<float>();
		}

		if ((updateNum % dqnOption.logInterval) == 0) {
//			auto lossValue = loss.item<float>();
//			auto qs = (curDist.softmax(-1) * valueItems).sum(-1, false).mean();
//			auto qsValue = qs.item<float>();

			tLogger.add_scalar("loss/entLoss", updateNum, updateLoss / (float)dqnOption.epochPerUpdate);
			tLogger.add_scalar("stat/qs", updateNum, updateQs / (float)dqnOption.epochPerUpdate);
		}

		//TEST
		if (dqnOption.toTest) {
			if ((updateNum % dqnOption.testGapEp) == 0) {
				test(dqnOption.testEp, false, false);
			}
		}
	}

	save();
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void CategoricalDqn<NetType, EnvType, PolicyType, OptimizerType>::updateModel(bool force) {
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
void CategoricalDqn<NetType, EnvType, PolicyType, OptimizerType>::updateStep(const float epochNum) {
	updateModel(false);

	if (updateNum > (dqnOption.explorePart * epochNum)) {
		return;
	}
	float newEpsilon = (dqnOption.exploreBegin - dqnOption.exploreEnd) * (epochNum * dqnOption.explorePart - updateNum) / (epochNum * dqnOption.explorePart) + dqnOption.exploreEnd;
	policy.updateEpsilon(newEpsilon);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void CategoricalDqn<NetType, EnvType, PolicyType, OptimizerType>::test(const int epochNum, bool render, bool toLoad) {
	if (toLoad) {
		load();
		updateModel();
	}

	tModel.eval();

	std::vector<long> testShapeData;
	testShapeData.push_back(dqnOption.testBatch);
	for (int i = 1; i < inputShape.size(); i ++) {
		testShapeData.push_back(inputShape[i]);
	}
	at::IntArrayRef testInputShape(testShapeData);
//	LOG4CXX_INFO(logger, "testInputShape: " << testInputShape);

	int updateEp = 0;
	float totalLen = 0;
	float totalReward = 0;

	std::vector<float> stateVec = testEnv.reset();
	while (updateEp < dqnOption.testEp) {
		torch::Tensor inputTensor = torch::from_blob(stateVec.data(), testInputShape).div(dqnOption.inputScale).to(deviceType);
		torch::Tensor outputTensor;
		{
			torch::NoGradGuard guard;
			outputTensor = tModel.forward(inputTensor).detach();
		}
		outputTensor = outputTensor.view({dqnOption.testBatch, dqnOption.outputNum, dqnOption.atomNum});
		outputTensor = outputTensor.softmax(-1);
		torch::Tensor qs = outputTensor * valueItems;
//		LOG4CXX_INFO(logger, "test input: " << inputTensor);
//		LOG4CXX_INFO(logger, "test qs: " << qs);
		qs = qs.sum(-1, false);//{batchSize, outputNum}
		std::vector<int64_t> actions = policy.getTestActions(qs);
//		LOG4CXX_INFO(logger, "actions: " << actions);

		auto stepResult = testEnv.step(actions, true);
		auto nextInputVec = std::get<0>(stepResult);
		auto rewardVec = std::get<1>(stepResult);
		auto doneVec = std::get<2>(stepResult);
//		LOG4CXX_INFO(logger, "rewardVec: " << rewardVec);

		Stats::UpdateReward(statRewards, rewardVec);
		Stats::UpdateLen(statLens);
		Stats::UpdateReward(statEpRewards, rewardVec);
		Stats::UpdateLen(statEpLens);


		for (int i = 0; i < dqnOption.testBatch; i ++) {
			if (doneVec[i]) {
//				totalLen += statLens[i];
//				totalReward += statRewards[i];

				LOG4CXX_INFO(logger, "ep" << totalTestEp << ": " << statRewards[i] << ", " << statLens[i]);
//				testStater.update(statLens[i], statRewards[i]);
				statRewards[i] = 0;
				statLens[i] = 0;

				livePerEp[i] ++;
				if (livePerEp[i] == dqnOption.livePerEpisode) {
					updateEp ++;
//					testEpNum ++;
					totalTestEp ++;
					totalLen += statEpLens[i];
					totalReward += statEpRewards[i];

					tLogger.add_scalar("test/reward", totalTestEp, statEpRewards[i]);
					tLogger.add_scalar("test/len", totalTestEp, statEpLens[i]);

					statEpRewards[i] = 0;
					statEpLens[i] = 0;
					livePerEp[i] = 0;
				}
			}
		}

		stateVec = nextInputVec;
	}

	float aveLen = totalLen / (float)updateEp;
	float aveReward = totalReward / (float)updateEp;
	tLogger.add_scalar("test/ave_reward", updateNum, aveReward);
	tLogger.add_scalar("test/ave_len", updateNum, aveLen);

	if (dqnOption.saveModel) {
		if (aveReward > maxTestReward) {
			maxTestReward = aveReward + dqnOption.saveStep;
			saveTModel(aveReward);
		}
	}
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void CategoricalDqn<NetType, EnvType, PolicyType, OptimizerType>::save() {
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
void CategoricalDqn<NetType, EnvType, PolicyType, OptimizerType>::saveTModel(float reward) {
	if (!dqnOption.saveModel) {
		return;
	}

	std::string modelPath = dqnOption.savePathPrefix + "_" + std::to_string(reward) + "_model" + ".pt";
	torch::serialize::OutputArchive outputArchive;
	tModel.save(outputArchive);
	outputArchive.save_to(modelPath);
	LOG4CXX_INFO(logger, "Save model into " << modelPath);

	std::string optPath = dqnOption.savePathPrefix + "_" + std::to_string(reward) + "_optimizer" + ".pt";
	torch::serialize::OutputArchive optimizerArchive;
	optimizer.save(optimizerArchive);
	optimizerArchive.save_to(optPath);
	LOG4CXX_INFO(logger, "Save optimizer into " << optPath);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void CategoricalDqn<NetType, EnvType, PolicyType, OptimizerType>::load() {
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


#endif /* INC_ALG_CATDQN_HPP_ */
