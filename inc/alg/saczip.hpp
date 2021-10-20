/*
 * saczip.hpp
 *
 *  Created on: Oct 1, 2021
 *      Author: zf
 */

#ifndef INC_ALG_SACZIP_HPP_
#define INC_ALG_SACZIP_HPP_



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

template<typename QNetType, typename PNetType, typename EnvType, typename PolicyType, typename QOptimizerType, typename POptimizerType, typename AlphaOptimizerType>
class SacZip {
private:
	QNetType& bModel1;
	QNetType& bModel2;
	QNetType& tModel1;
	QNetType& tModel2;
	QOptimizerType& qOptimizer1;
	QOptimizerType& qOptimizer2;

	PNetType& policyModel;
	POptimizerType& pOptimizer;

	torch::Tensor& logAlpha;
	torch::Tensor alpha;
//	torch::Tensor targetEntropy;
	float targetEntropy;
	int entropyIndex = 0;
	AlphaOptimizerType& alphaOptimizer;

	EnvType& env;
	PolicyType& policy;
	const torch::Device deviceType;
	const at::IntArrayRef inputShape;
//	at::IntArrayRef testInputShape;

	EnvType& testEnv;

	const DqnOption dqnOption;

	uint32_t updateNum = 0;
	uint32_t totalTestEp = 0;

	float maxReward;

//	const int actionNum;
//	std::vector<int64_t> indice;

	const torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("sac");
	TensorBoardLogger tLogger;


//	Stats stater;
//	Stats testStater;
//	LossStats lossStater;


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
//		torch::Tensor nextStates;
		torch::Tensor actions;
		torch::Tensor rewards;
		torch::Tensor donesMask;

		//Store states and rewards after normalization
		void add(torch::Tensor state, torch::Tensor nextState, int action, float reward, float done);
		torch::Tensor getSampleIndex(int batchSize);
	};

	ReplayBuffer buffer; //buffer has to be defined after dqnOption so ReplayBuffer can get all parameters of dqnOption.

	void updateModel(bool force = false);
	void disableTargetGrad();

	void load();
	void save(std::string flag = "");

	void testModel(const int epochNum, bool render = false);
public:
	SacZip(QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
			PNetType& pModel, POptimizerType& pOpt,
			torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
			EnvType& iEnv, PolicyType& iPolicy,
			EnvType& testEnv,
			const torch::Device dType, DqnOption iOption);
	~SacZip() = default;
	SacZip(const SacZip&) = delete;

	void train(const int epochNum);
	void test(const int epochNum, bool render = false);
};

template <typename QNetType, typename PNetType, typename EnvType, typename PolicyType, typename QOptimizerType, typename POptimizerType, typename AlphaOptimizerType>
SacZip<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::ReplayBuffer::ReplayBuffer(const int iCap, const at::IntArrayRef& inputShape): cap(iCap) {
	std::vector<int64_t> stateInputShape;
	stateInputShape.push_back(cap);
	//input state shape = {1, 4, 84, 84};
	for (int i = 1; i < inputShape.size(); i ++) {
		stateInputShape.push_back(inputShape[i]);
	}
	at::IntArrayRef outputShape{ReplayBuffer::cap, 1};

	states = torch::zeros(stateInputShape, byteOpt);
//	states = torch::zeros(stateInputShape);
//	nextStates = torch::zeros(stateInputShape, byteOpt);
	actions = torch::zeros(outputShape, byteOpt);
	rewards = torch::zeros(outputShape);
	donesMask = torch::zeros(outputShape, byteOpt);

	LOG4CXX_DEBUG(logger, "Replay buffer ready");
}

template <typename QNetType, typename PNetType, typename EnvType, typename PolicyType, typename QOptimizerType, typename POptimizerType, typename AlphaOptimizerType>
void SacZip<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::ReplayBuffer::add(
		torch::Tensor state, torch::Tensor nextState, int action, float reward, float done) {
//	{
//		//For log
//		bool isSame = states[curIndex].equal(state);
//		LOG4CXX_DEBUG(logger, curIndex << ": The state same? " << isSame);
//		if (!isSame) {
//			LOG4CXX_DEBUG(logger, "curState: " << states[curIndex]);
//			LOG4CXX_DEBUG(logger, "inputState: " << state);
//			LOG4CXX_DEBUG(logger, "nextState: " << nextState);
//		}
//	}

	int nextIndex = (curIndex + 1) % cap;

	torch::Tensor inputState = state.to(torch::kByte);
	torch::Tensor inputNextState = nextState.to(torch::kByte);
//	auto inputState = state;
//	auto inputNextState = nextState;

//	{
//		//For log
//		bool isSame = states[curIndex].equal(inputState);
//		LOG4CXX_DEBUG(logger, curIndex << ": The state same? " << isSame);
//		if (!isSame) {
//			LOG4CXX_DEBUG(logger, "curState: " << states[curIndex]);
//			LOG4CXX_DEBUG(logger, "inputState: " << inputState);
//			LOG4CXX_DEBUG(logger, "nextState: " << inputNextState);
//		}
//	}

	states[curIndex].copy_(inputState.squeeze());
	states[nextIndex].copy_(inputNextState.squeeze()); //TODO: Optimize
//	nextStates[curIndex].copy_(inputNextState.squeeze());
	actions[curIndex][0] = action;
	rewards[curIndex][0] = reward;
	donesMask[curIndex][0] = done;
	LOG4CXX_DEBUG(logger, "states after copy: " << states[curIndex]);

	curIndex = nextIndex;
	if (curSize < cap) {
		curSize ++;
	}
}

template <typename QNetType, typename PNetType, typename EnvType, typename PolicyType, typename QOptimizerType, typename POptimizerType, typename AlphaOptimizerType>
torch::Tensor SacZip<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::ReplayBuffer::getSampleIndex(int batchSize) {
	torch::Tensor indices = torch::randint(0, curSize - 1, {batchSize}, longOpt);

	indices = (indices + (curIndex + 1)) % curSize; //No curIndex involved as its next not match

	return indices;
}


template <typename QNetType, typename PNetType, typename EnvType, typename PolicyType, typename QOptimizerType, typename POptimizerType, typename AlphaOptimizerType>
SacZip<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::SacZip(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType& iPolicy,
		EnvType& tEnv,
		const torch::Device dType, DqnOption iOption):
	bModel1(qModel1),
	bModel2(qModel2),
	tModel1(qTargetModel1),
	tModel2(qTargetModel2),
	qOptimizer1(qOpt1),
	qOptimizer2(qOpt2),
	policyModel(pModel),
	pOptimizer(pOpt),
	logAlpha(iLogAlpha),
	alphaOptimizer(aOpt),
	env(iEnv),
	policy(iPolicy),
	dqnOption(iOption),
	testEnv(tEnv),
	deviceType(iOption.deviceType),
	inputShape(iOption.inputShape),
	buffer(iOption.rbCap, iOption.inputShape),
	maxReward(iOption.saveThreshold),
//	stater(iOption.statPathPrefix + "_stat.txt", iOption.statCap),
//	testStater(iOption.statPathPrefix + "_test.txt", iOption.testEp),
//	lossStater(iOption.statPathPrefix + "_loss.txt"),
	tLogger(iOption.tensorboardLogPath.c_str())

//	testLogAlpha(torch::zeros(1, torch::TensorOptions().requires_grad(true).device(iOption.deviceType))),
//	testOptA({testLogAlpha}, torch::optim::AdamOptions(0.0003))
{
//	logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType)) * std::log(0.2);
	alpha = logAlpha.exp().detach();
	disableTargetGrad();
//	testLogAlpha = torch::zeros(1, torch::TensorOptions().requires_grad(true).device(deviceType));
//	testAlpha = testLogAlpha.exp();
//	testOptA = torch::optim::Adam({testLogAlpha}, torch::optim::AdamOptions(0.0003));

	//TODO: Initialize targetEntropy
//	targetEntropy = torch::ones({1}, torch::TensorOptions().device(deviceType)) * dqnOption.targetEntropy;
	if (dqnOption.fixedEntropy) {
		targetEntropy = dqnOption.targetEntropy;
	} else {
		targetEntropy = dqnOption.targetEntropies[entropyIndex];
		entropyIndex ++;
	}
	LOG4CXX_INFO(logger, "targetEntropy: " << dqnOption.targetEntropy);
}

template <typename QNetType, typename PNetType, typename EnvType, typename PolicyType, typename QOptimizerType, typename POptimizerType, typename AlphaOptimizerType>
void SacZip<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::train(const int epochNum) {
	load();
	updateModel(true); //model assignment
//	tModel1.eval();
//	tModel2.eval();

	std::vector<float> stateVec = env.reset();

	//only one env
	std::vector<float> statRewards(dqnOption.envNum, 0);
	std::vector<float> statLens(dqnOption.envNum, 0);

	while (updateNum < epochNum) {
		for (int i = 0; i < dqnOption.envStep; i ++) {
			updateNum ++;
			//Run step
			torch::Tensor cpuinputTensor = torch::from_blob(stateVec.data(), inputShape); //.div(dqnOption.inputScale);
			torch::Tensor inputTensor = cpuinputTensor.div(dqnOption.inputScale).to(deviceType);

			torch::Tensor actionSamples;
			if (updateNum < dqnOption.startStep) {
				actionSamples = torch::randint(0, dqnOption.outputNum, {dqnOption.envNum}, longOpt);
			} else {
				torch::NoGradGuard guard;
				torch::Tensor actionOutput = policyModel.forward(inputTensor).detach();
				torch::Tensor actionProbs = torch::softmax(actionOutput, -1);
				//			std::vector<int64_t> actions = policy.getActions(actionProbs);
				actionSamples = actionProbs.multinomial(1).to(torch::kCPU);
			}
			std::vector<int64_t> actions(actionSamples.data_ptr<int64_t>(), actionSamples.data_ptr<int64_t>() + dqnOption.envNum);

			auto stepResult = env.step(actions, false);
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
				LOG4CXX_INFO(logger, "--" << updateNum << ": " << statRewards[0] << ", " << statLens[0]);
//				stater.update(statLens[0], statRewards[0]);
				statRewards[0] = 0;
				statLens[0] = 0;
			}

			torch::Tensor nextInputTensor = torch::from_blob(nextInputVec.data(), inputShape); //.div(dqnOption.inputScale);
			float reward = std::max(std::min((rewardVec[0] * dqnOption.rewardScale), dqnOption.rewardMax), dqnOption.rewardMin);
			buffer.add(cpuinputTensor, nextInputTensor, actions[0], reward, doneMask);

			stateVec = nextInputVec;
		}

		if (updateNum < dqnOption.startStep) {
			continue;
		}

		if (!dqnOption.fixedEntropy) {
			if (entropyIndex < dqnOption.targetEntropies.size()) {
				if (updateNum >= dqnOption.targetSteps[entropyIndex]) {
					targetEntropy = dqnOption.targetEntropies[entropyIndex];
					entropyIndex ++;
				}
			}
		}

		/*
		 * Learning
		 */

		//Sample
		torch::Tensor sampleIndice = buffer.getSampleIndex(dqnOption.batchSize);
		torch::Tensor curStateTensor = buffer.states.index_select(0, sampleIndice).to(deviceType);
		torch::Tensor actionTensor = buffer.actions.index_select(0, sampleIndice).to(deviceType);
		torch::Tensor rewardTensor = buffer.rewards.index_select(0, sampleIndice).to(deviceType);
		torch::Tensor doneMaskTensor = buffer.donesMask.index_select(0, sampleIndice).to(deviceType);
//		torch::Tensor nextStateTensor = buffer.nextStates.index_select(0, sampleIndice).to(deviceType);
		auto nextSampleIndice = (sampleIndice + 1) % dqnOption.rbCap;
		torch::Tensor nextStateTensor = buffer.states.index_select(0, nextSampleIndice).to(deviceType);

		curStateTensor = curStateTensor.to(torch::kFloat).div(dqnOption.inputScale);
		nextStateTensor = nextStateTensor.to(torch::kFloat).div(dqnOption.inputScale);
		rewardTensor = rewardTensor.to(torch::kFloat);
		doneMaskTensor = doneMaskTensor.to(torch::kFloat);
		actionTensor = actionTensor.to(torch::kLong);

		//Q loss
		torch::Tensor targetQ;
		{
			torch::NoGradGuard guard;

			torch::Tensor nextProbOutput = policyModel.forward(nextStateTensor);
			torch::Tensor nextProbs = torch::softmax(nextProbOutput, -1);
			torch::Tensor nextLogProbs = torch::log_softmax(nextProbOutput, -1);
			torch::Tensor nextQ1 = tModel1.forward(nextStateTensor);
			torch::Tensor nextQ2 = tModel2.forward(nextStateTensor);
			torch::Tensor nextQ = torch::min(nextQ1, nextQ2);

			assert(nextProbs.sizes() == nextQ.sizes());
			torch::Tensor nextV = (nextProbs * (nextQ - alpha * nextLogProbs)).sum(-1, true);
			assert(rewardTensor.sizes() == nextV.sizes());
			assert(nextV.sizes() == doneMaskTensor.sizes());
			targetQ = rewardTensor + dqnOption.gamma * doneMaskTensor * nextV;
			targetQ = targetQ.detach();
		}
		torch::Tensor curQ1 = bModel1.forward(curStateTensor);
		torch::Tensor curQ2 = bModel2.forward(curStateTensor);
		torch::Tensor q1 = curQ1.gather(-1, actionTensor);
		torch::Tensor q2 = curQ2.gather(-1, actionTensor);

		assert(q1.sizes() == targetQ.sizes());
		assert(q2.sizes() == targetQ.sizes());
		torch::Tensor qLoss1 = torch::nn::functional::mse_loss(q1, targetQ);
		torch::Tensor qLoss2 = torch::nn::functional::mse_loss(q2, targetQ);

		//Policy loss
		torch::Tensor curQ;
		{
			torch::NoGradGuard guard;
			auto curQ1 = bModel1.forward(curStateTensor);
			auto curQ2 = bModel2.forward(curStateTensor);
			curQ = torch::min(curQ1, curQ2);
			curQ = curQ.detach();
		}
		torch::Tensor curProbOutput = policyModel.forward(curStateTensor);
		torch::Tensor curProbs = torch::softmax(curProbOutput, -1);
		torch::Tensor curLogProbs = torch::log_softmax(curProbOutput, -1);
		torch::Tensor policyEntropy = - torch::sum(curProbs * curLogProbs, 1, true);
		torch::Tensor qDist = torch::sum(curQ * curProbs, 1, true);
		torch::Tensor policyLoss = (-qDist - alpha * policyEntropy).mean();


		//Alpha loss
		torch::Tensor alphaLoss = - torch::mean(logAlpha * (targetEntropy - policyEntropy.detach()));


		/*
		 * UPDATE
		 */
		qOptimizer1.zero_grad();
		qLoss1.backward();
//		torch::nn::utils::clip_grad_norm_(bModel1.parameters(), dqnOption.maxGradNormClip);
		qOptimizer1.step();

		qOptimizer2.zero_grad();
		qLoss2.backward();
//		torch::nn::utils::clip_grad_norm_(bModel2.parameters(), dqnOption.maxGradNormClip);
		qOptimizer2.step();

		pOptimizer.zero_grad();
		policyLoss.backward();
		pOptimizer.step();


		alphaOptimizer.zero_grad();
		alphaLoss.backward();
		alphaOptimizer.step();

		//log
		if ((updateNum % dqnOption.logInterval) == 0) {
			float qLoss1Value = qLoss1.item<float>();
			float qLoss2Value = qLoss2.item<float>();
			float policyLossValue = policyLoss.item<float>();
			float alphaLossValue = alphaLoss.item<float>();
			float q1Value = q1.mean().item<float>();
			float q2Value = q2.mean().item<float>();
			float entropyValue = policyEntropy.mean().abs().item<float>();
			float alphaValue = alpha.item<float>();
			float vValue = qDist.mean().item<float>();

			tLogger.add_scalar("loss/q1", updateNum, qLoss1Value);
			tLogger.add_scalar("loss/q2", updateNum, qLoss2Value);
			tLogger.add_scalar("loss/policy", updateNum, policyLossValue);
			tLogger.add_scalar("loss/alpha", updateNum, alphaLossValue);

			tLogger.add_scalar("stat/q1", updateNum, q1Value);
			tLogger.add_scalar("stat/q2", updateNum, q2Value);
			tLogger.add_scalar("stat/entropy", updateNum, entropyValue);
			tLogger.add_scalar("stat/alpha", updateNum, alphaValue);
			tLogger.add_scalar("stat/V", updateNum, vValue);
		}

		alpha = logAlpha.exp().detach();
//		alpha = torch::clamp(logAlpha.exp(), dqnOption.targetEntropy, 1);
//		alpha = torch::clamp(logAlpha.exp(), 0, 1);
//
//		if ((updateNum % dqnOption.logInterval) == 0) {
//			lossStater.update({alpha.item<float>(), entropyValue, policyLossValue, alphaLossValue});
//		}

		updateModel(false);

		if ((updateNum % dqnOption.testGapEp) == 0) {
			if (dqnOption.toTest) {
				testModel(dqnOption.testEp, dqnOption.testRender);
			}
		}
	}

	save();
}

template <typename QNetType, typename PNetType, typename EnvType, typename PolicyType, typename QOptimizerType, typename POptimizerType, typename AlphaOptimizerType>
void SacZip<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::disableTargetGrad() {
	auto params1 = tModel1.parameters();
	for (auto& param: params1) {
		param.requires_grad_(false);
	}

	auto params2 = tModel2.parameters();
	for (auto& param: params2) {
		param.requires_grad_(false);
	}
}

template <typename QNetType, typename PNetType, typename EnvType, typename PolicyType, typename QOptimizerType, typename POptimizerType, typename AlphaOptimizerType>
void SacZip<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::updateModel(bool force) {
	if (!force) {
		if ((updateNum % dqnOption.targetUpdateStep) != 0) {
			return;
		}
	}

	torch::NoGradGuard guard;

	auto paramDict1 = bModel1.named_parameters();
	auto buffDict1 = bModel1.named_buffers();
	auto targetParamDict1 = tModel1.named_parameters();
	auto targetBuffDict1 = tModel1.named_buffers();

	for (const auto& item: paramDict1) {
		const auto& key = item.key();
		const auto param = item.value();
		auto& targetParam = targetParamDict1[key];

		targetParam.mul_(1 - dqnOption.tau);
		targetParam.add_(param, dqnOption.tau);
	}

	for (const auto& item: buffDict1) {
		const auto& key = item.key();
		const auto& buff = item.value();
		auto& targetBuff = targetBuffDict1[key];

		targetBuff.mul(1 - dqnOption.tau);
		targetBuff.add_(buff, dqnOption.tau);
	}
	LOG4CXX_INFO(logger, "target network 1 synched");

	auto paramDict2 = bModel2.named_parameters();
	auto buffDict2 = bModel2.named_buffers();
	auto targetParamDict2 = tModel2.named_parameters();
	auto targetBuffDict2 = tModel2.named_buffers();

	for (const auto& item: paramDict2) {
		const auto& key = item.key();
		const auto param = item.value();
		auto& targetParam = targetParamDict2[key];

		targetParam.mul_(1 - dqnOption.tau);
		targetParam.add_(param, dqnOption.tau);
	}

	for (const auto& item: buffDict2) {
		const auto& key = item.key();
		const auto& buff = item.value();
		auto& targetBuff = targetBuffDict2[key];

		targetBuff.mul(1 - dqnOption.tau);
		targetBuff.add_(buff, dqnOption.tau);
	}
	LOG4CXX_INFO(logger, "target network 2 synched");
}

template <typename QNetType, typename PNetType, typename EnvType, typename PolicyType, typename QOptimizerType, typename POptimizerType, typename AlphaOptimizerType>
void SacZip<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::testModel(const int epochNum, bool render) {
//	policyModel.eval();

	std::vector<float> statRewards(dqnOption.testBatch, 0);
	std::vector<float> statLens(dqnOption.testBatch, 0);
	std::vector<float> statEpRewards(dqnOption.testBatch, 0);
	std::vector<float> statEpLens(dqnOption.testBatch, 0);
	std::vector<int> livePerEp(dqnOption.testBatch, 0);

	std::vector<long> testShapeData;
	testShapeData.push_back(dqnOption.testBatch);
	for (int i = 1; i < inputShape.size(); i ++) {
		testShapeData.push_back(inputShape[i]);
	}
	at::IntArrayRef testInputShape(testShapeData);
//	LOG4CXX_INFO(logger, "testInputShape: " << testInputShape);

	std::vector<float> stateVec = testEnv.reset();
//	int testStep = 0;
	int testEpNum = 0;
	int updateEp = 0;
	float totalLen = 0;
	float totalReward = 0;

	while (updateEp < dqnOption.testEp) {
//	while (testStep < dqnOption.testEp) {
//		testStep ++;

//		std::vector<int64_t> actions(dqnOption.testBatch, 0);
		torch::Tensor greedyOutput;
		{
			torch::NoGradGuard guard;
			torch::Tensor inputTensor = torch::from_blob(stateVec.data(), testInputShape).div(dqnOption.inputScale).to(deviceType);

			torch::Tensor outputTensor = policyModel.forward(inputTensor).detach();
//			greedyOutput = torch::softmax(outputTensor, -1);
			greedyOutput = outputTensor.argmax(-1, true).to(torch::kCPU);
			//		LOG4CXX_INFO(logger, "greedyOutput: " << greedyOutput);
//			int64_t* greedyPtr = greedyOutput.data_ptr<int64_t>();
//			for (int i = 0; i < dqnOption.testBatch; i ++) {
//				actions[i] = greedyPtr[i];
//			}
		}
		std::vector<int64_t> actions(greedyOutput.data_ptr<int64_t>(), greedyOutput.data_ptr<int64_t>() + dqnOption.testBatch);
//		auto actions = policy.getActions(greedyOutput);


//		torch::Tensor actionProbs = torch::softmax(outputTensor, -1);
//		std::vector<int64_t> actions = policy.getTestActions(actionProbs);

		auto stepResult = testEnv.step(actions, render);
		auto nextInputVec = std::get<0>(stepResult);
		auto rewardVec = std::get<1>(stepResult);
		auto doneVec = std::get<2>(stepResult);

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
					testEpNum ++;
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

	float aveLen = totalLen / (float)testEpNum;
	float aveReward = totalReward / (float)testEpNum;
	tLogger.add_scalar("test/ave_reward", updateNum, aveReward);
	tLogger.add_scalar("test/ave_len", updateNum, aveLen);

	if (aveReward > maxReward) {
		save("_" + std::to_string(aveReward));
		maxReward = aveReward + dqnOption.saveStep;
	}

//	policyModel.train();
}


template <typename QNetType, typename PNetType, typename EnvType, typename PolicyType, typename QOptimizerType, typename POptimizerType, typename AlphaOptimizerType>
void SacZip<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::test(const int epochNum, bool render) {
	load();

	testModel(epochNum, render);
}

template <typename QNetType, typename PNetType, typename EnvType, typename PolicyType, typename QOptimizerType, typename POptimizerType, typename AlphaOptimizerType>
void SacZip<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::save(std::string flag) {
	if (!dqnOption.saveModel) {
		return;
	}

	std::string modelPathQ1 = dqnOption.savePathPrefix + flag + "_q1_model.pt";
	torch::serialize::OutputArchive outputArchiveQ1;
	bModel1.save(outputArchiveQ1);
	outputArchiveQ1.save_to(modelPathQ1);
	LOG4CXX_INFO(logger, "Save model into " << modelPathQ1);

	std::string modelPathQ2 = dqnOption.savePathPrefix + flag + "_q2_model.pt";
	torch::serialize::OutputArchive outputArchiveQ2;
	bModel2.save(outputArchiveQ2);
	outputArchiveQ2.save_to(modelPathQ2);
	LOG4CXX_INFO(logger, "Save model into " << modelPathQ2);

	std::string optPathQ1 = dqnOption.savePathPrefix + flag + "_q1_optimizer.pt";
	torch::serialize::OutputArchive optimizerArchiveQ1;
	qOptimizer1.save(optimizerArchiveQ1);
	optimizerArchiveQ1.save_to(optPathQ1);
	LOG4CXX_INFO(logger, "Save optimizer into " << optPathQ1);

	std::string optPathQ2 = dqnOption.savePathPrefix + flag + "_q2_optimizer.pt";
	torch::serialize::OutputArchive optimizerArchiveQ2;
	qOptimizer2.save(optimizerArchiveQ2);
	optimizerArchiveQ2.save_to(optPathQ2);
	LOG4CXX_INFO(logger, "Save optimizer into " << optPathQ2);

	std::string modelPathP = dqnOption.savePathPrefix + flag + "_p_model.pt";
	torch::serialize::OutputArchive outputArchiveP;
	policyModel.save(outputArchiveP);
	outputArchiveP.save_to(modelPathP);
	LOG4CXX_INFO(logger, "Save model into " << modelPathP);

	std::string optPathP = dqnOption.savePathPrefix + flag + "_p_optimizer.pt";
	torch::serialize::OutputArchive optimizerArchiveP;
	pOptimizer.save(optimizerArchiveP);
	optimizerArchiveP.save_to(optPathP);
	LOG4CXX_INFO(logger, "Save optimizer into " << optPathP);

	std::string alphaPath = dqnOption.savePathPrefix + flag + "_alpha.pt";
	torch::save(logAlpha, alphaPath);
	LOG4CXX_INFO(logger, "Save alpha into " << alphaPath);

	std::string optPathA = dqnOption.savePathPrefix + flag + "_a_optimizer.pt";
	torch::serialize::OutputArchive optimizerArchiveA;
	alphaOptimizer.save(optimizerArchiveA);
	optimizerArchiveA.save_to(optPathA);
	LOG4CXX_INFO(logger, "Save optimizer into " << optPathA);
}


template <typename QNetType, typename PNetType, typename EnvType, typename PolicyType, typename QOptimizerType, typename POptimizerType, typename AlphaOptimizerType>
void SacZip<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::load() {
	if (!dqnOption.loadModel) {
		return;
	}

	std::string modelPathQ1 = dqnOption.loadPathPrefix + "_q1_model.pt";
	torch::serialize::InputArchive inChiveQ1;
	inChiveQ1.load_from(modelPathQ1);
	bModel1.load(inChiveQ1);
	LOG4CXX_INFO(logger, "Load model from " << modelPathQ1);

	std::string modelPathQ2 = dqnOption.loadPathPrefix + "_q2_model.pt";
	torch::serialize::InputArchive inChiveQ2;
	inChiveQ2.load_from(modelPathQ2);
	bModel2.load(inChiveQ2);
	LOG4CXX_INFO(logger, "Load model from " << modelPathQ2);

	std::string modelPathP = dqnOption.loadPathPrefix + "_p_model.pt";
	torch::serialize::InputArchive inChiveP;
	inChiveP.load_from(modelPathP);
	policyModel.load(inChiveQ2);
	LOG4CXX_INFO(logger, "Load model from " << modelPathP);

	std::string modelPathA = dqnOption.loadPathPrefix + "_alpha.pt";
//	torch::Tensor tmpLogAlpha = torch::load(modelPathA);
//	logAlpha.copy_(tmpLogAlpha);
	torch::load(logAlpha, modelPathA);
	alpha = logAlpha.exp().detach();
	LOG4CXX_INFO(logger, "Load model from " << modelPathA);
	LOG4CXX_INFO(logger, "alpha value: " << logAlpha.exp());


	if (dqnOption.loadOptimizer) {
		std::string optPathQ1 = dqnOption.loadPathPrefix + "_q1_optimizer.pt";
		torch::serialize::InputArchive opInChiveQ1;
		opInChiveQ1.load_from(optPathQ1);
		qOptimizer1.load(opInChiveQ1);
		LOG4CXX_INFO(logger, "Load optimizer from " << optPathQ1);

		std::string optPathQ2 = dqnOption.loadPathPrefix + "_q2_optimizer.pt";
		torch::serialize::InputArchive opInChiveQ2;
		opInChiveQ2.load_from(optPathQ2);
		qOptimizer2.load(opInChiveQ2);
		LOG4CXX_INFO(logger, "Load optimizer from " << optPathQ2);

		std::string optPathP = dqnOption.loadPathPrefix + "_p_optimizer.pt";
		torch::serialize::InputArchive opInChiveP;
		opInChiveP.load_from(optPathP);
		pOptimizer.load(opInChiveP);
		LOG4CXX_INFO(logger, "Load optimizer from " << optPathP);

		std::string optPathA = dqnOption.loadPathPrefix + "_a_optimizer.pt";
		torch::serialize::InputArchive opInChiveA;
		opInChiveA.load_from(optPathA);
		alphaOptimizer.load(opInChiveA);
		LOG4CXX_INFO(logger, "Load optimizer from " << optPathA);
	}

}




#endif /* INC_ALG_SACZIP_HPP_ */
