/*
 * sac.hpp
 *
 *  Created on: Sep 16, 2021
 *      Author: zf
 */

#ifndef INC_ALG_SAC_HPP_
#define INC_ALG_SAC_HPP_



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
#include "alg/utils/dqnoption.h"

template<typename QNetType, typename PNetType, typename EnvType, typename PolicyType, typename QOptimizerType, typename POptimizerType, typename AlphaOptimizerType>
class Sac {
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
	torch::Tensor targetEntropy;
	AlphaOptimizerType& alphaOptimizer;

	EnvType& env;
	PolicyType& policy;
	const torch::Device deviceType;
	const at::IntArrayRef inputShape;

	EnvType& testEnv;

	const DqnOption dqnOption;

	uint32_t updateNum = 0;

//	const int actionNum;
//	std::vector<int64_t> indice;

	const torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("sac");

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

	void load();
	void save();

	void testModel(const int epochNum, bool render = false);
public:
	Sac(QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
			PNetType& pModel, POptimizerType& pOpt,
			torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
			EnvType& iEnv, PolicyType& iPolicy,
			EnvType& testEnv,
			const torch::Device dType, DqnOption iOption);
	~Sac() = default;
	Sac(const Sac&) = delete;

	void train(const int epochNum);
	void test(const int epochNum, bool render = false);
};

template <typename QNetType, typename PNetType, typename EnvType, typename PolicyType, typename QOptimizerType, typename POptimizerType, typename AlphaOptimizerType>
Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::ReplayBuffer::ReplayBuffer(const int iCap, const at::IntArrayRef& inputShape): cap(iCap) {
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

template <typename QNetType, typename PNetType, typename EnvType, typename PolicyType, typename QOptimizerType, typename POptimizerType, typename AlphaOptimizerType>
void Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::ReplayBuffer::add(
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

template <typename QNetType, typename PNetType, typename EnvType, typename PolicyType, typename QOptimizerType, typename POptimizerType, typename AlphaOptimizerType>
torch::Tensor Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::ReplayBuffer::getSampleIndex(int batchSize) {
	torch::Tensor indices = torch::randint(0, curSize, {batchSize}, longOpt);

	return indices;
}


template <typename QNetType, typename PNetType, typename EnvType, typename PolicyType, typename QOptimizerType, typename POptimizerType, typename AlphaOptimizerType>
Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
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
	stater(iOption.statPathPrefix + "_stat.txt", iOption.statCap),
	testStater(iOption.statPathPrefix + "_test.txt", iOption.statCap),
	lossStater(iOption.statPathPrefix + "_loss.txt")

//	testLogAlpha(torch::zeros(1, torch::TensorOptions().requires_grad(true).device(iOption.deviceType))),
//	testOptA({testLogAlpha}, torch::optim::AdamOptions(0.0003))
{
//	logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType)) * std::log(0.2);
	alpha = logAlpha.exp();
//	testLogAlpha = torch::zeros(1, torch::TensorOptions().requires_grad(true).device(deviceType));
//	testAlpha = testLogAlpha.exp();
//	testOptA = torch::optim::Adam({testLogAlpha}, torch::optim::AdamOptions(0.0003));

	//TODO: Initialize targetEntropy
//	targetEntropy = torch::ones({1}, torch::TensorOptions().device(deviceType)) * dqnOption.targetEntropy;
	LOG4CXX_INFO(logger, "targetEntropy: " << dqnOption.targetEntropy);
}

template <typename QNetType, typename PNetType, typename EnvType, typename PolicyType, typename QOptimizerType, typename POptimizerType, typename AlphaOptimizerType>
void Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::train(const int epochNum) {
	load();
	updateModel(true); //model assignment
	tModel1.eval();
	tModel2.eval();

	std::vector<float> stateVec = env.reset();
//	std::vector<float> nextStateVec;

	//only one env
	std::vector<float> statRewards(dqnOption.envNum, 0);
	std::vector<float> statLens(dqnOption.envNum, 0);

	while (updateNum < epochNum) {
//		LOG4CXX_INFO(logger, "-------------------------------> Update " << updateNum);
		for (int i = 0; i < dqnOption.envStep; i ++) {
			updateNum ++;
			//Run step
			torch::Tensor cpuinputTensor = torch::from_blob(stateVec.data(), inputShape).div(dqnOption.inputScale);
			torch::Tensor inputTensor = cpuinputTensor.to(deviceType);

			torch::Tensor actionOutput = policyModel.forward(inputTensor);
			torch::Tensor actionProbs = torch::softmax(actionOutput, -1);
			//		LOG4CXX_INFO(logger, "actionProbs: " << actionProbs);
			std::vector<int64_t> actions = policy.getActions(actionProbs);

			auto stepResult = env.step(actions, false);
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
				LOG4CXX_INFO(logger, "--" << updateNum << stater);
			}

			torch::Tensor nextInputTensor = torch::from_blob(nextInputVec.data(), inputShape).div(dqnOption.inputScale);
			float reward = std::max(std::min((rewardVec[0] / dqnOption.rewardScale), dqnOption.rewardMax), dqnOption.rewardMin);
			buffer.add(cpuinputTensor, nextInputTensor, actions[0], reward, doneMask);

			stateVec = nextInputVec;
		}

		if (updateNum < dqnOption.startStep) {
			continue;
		}
		updateModel(false);

		/*
		 * Learning
		 */
		//Sample
		torch::Tensor sampleIndice = buffer.getSampleIndex(dqnOption.batchSize);
		LOG4CXX_DEBUG(logger, "sampleIndice: " << sampleIndice);
		torch::Tensor curStateTensor = buffer.states.index_select(0, sampleIndice).to(deviceType);
		torch::Tensor actionTensor = buffer.actions.index_select(0, sampleIndice).to(deviceType);
		torch::Tensor rewardTensor = buffer.rewards.index_select(0, sampleIndice).to(deviceType);
		torch::Tensor doneMaskTensor = buffer.donesMask.index_select(0, sampleIndice).to(deviceType);
		LOG4CXX_DEBUG(logger, "sampleIndex before: " << sampleIndice);
		sampleIndice = (sampleIndice + 1) % dqnOption.rbCap;
		torch::Tensor nextStateTensor = buffer.states.index_select(0, sampleIndice).to(deviceType);
		LOG4CXX_DEBUG(logger, "sampleIndice after: " << sampleIndice);
		LOG4CXX_DEBUG(logger, "nextStateTensor: " << nextStateTensor);
//		LOG4CXX_INFO(logger, "curStateTensor: " << curStateTensor);

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
			torch::Tensor nextV = (nextProbs * (nextQ - alpha * nextLogProbs)).sum(-1, true);
			targetQ = rewardTensor + dqnOption.gamma * doneMaskTensor * nextV;

//			torch::Tensor nextEntropy = ((-1) * nextProbs * nextLogProbs).sum(-1);
//			torch::Tensor aNextEntropy = alpha * nextEntropy;
//			LOG4CXX_INFO(logger, "nextEntropy: " << nextEntropy);
//			LOG4CXX_INFO(logger, "ratioNextEntropy: " << aNextEntropy);
//			LOG4CXX_INFO(logger, "reward: " << rewardTensor);
//			LOG4CXX_INFO(logger, "nextProbs: " << nextProbs);
//			LOG4CXX_INFO(logger, "nextLogProbs: " << nextLogProbs);
//			LOG4CXX_INFO(logger, "doneMasks: " << doneMaskTensor);
//			LOG4CXX_INFO(logger, "nextV: " << nextV);
//			LOG4CXX_INFO(logger, "targetQ: " << targetQ);
		}
		torch::Tensor curQ1 = bModel1.forward(curStateTensor);
		torch::Tensor curQ2 = bModel2.forward(curStateTensor);
//		LOG4CXX_INFO(logger, "curQ1: " << curQ1);
//		LOG4CXX_INFO(logger, "curQ2: " << curQ2);
		torch::Tensor q1 = curQ1.gather(-1, actionTensor);
		torch::Tensor q2 = curQ2.gather(-1, actionTensor);
		torch::Tensor qLoss1 = torch::nn::functional::mse_loss(q1, targetQ);
		torch::Tensor qLoss2 = torch::nn::functional::mse_loss(q2, targetQ);
//		LOG4CXX_INFO(logger, "actions: " << actionTensor);
//		LOG4CXX_INFO(logger, "q1: " << q1);

//		qOptimizer1.zero_grad();
//		qLoss1.backward();
//		qOptimizer1.step();
//		qOptimizer2.zero_grad();
//		qLoss2.backward();
//		qOptimizer2.step();

		//Policy loss
		torch::Tensor curQ;
		{
			torch::NoGradGuard guard;
			auto curQ1 = bModel1.forward(curStateTensor);
			auto curQ2 = bModel2.forward(curStateTensor);
			curQ = torch::min(curQ1, curQ2);
//			curQ = curQ2;
		}
		torch::Tensor curProbOutput = policyModel.forward(curStateTensor);
		torch::Tensor curProbs = torch::softmax(curProbOutput, -1);
		torch::Tensor curLogProbs = torch::log_softmax(curProbOutput, -1);
//		torch::Tensor tmp1 = alpha.detach() * curLogProbs;
//		torch::Tensor tmp2 = tmp1 - curQ;
//		torch::Tensor tmp3 = curProbs * tmp2;
//		torch::Tensor tmp4 = tmp3.sum(-1);
//		LOG4CXX_INFO(logger, "tmp1: " << tmp1);
//		LOG4CXX_INFO(logger, "tmp2: " << tmp2);
//		LOG4CXX_INFO(logger, "tmp3: " << tmp3);
//		LOG4CXX_INFO(logger, "tmp4: " << tmp4);
//		LOG4CXX_INFO(logger, "curProbs: " << curProbs);
//		LOG4CXX_INFO(logger, "curLogProbs: " << curLogProbs);
//		LOG4CXX_INFO(logger, "policyLoss: " << policyLoss);
//		LOG4CXX_INFO(logger, "curProbOutput " << curProbOutput);
		torch::Tensor policyEntropy = - torch::sum(curProbs * curLogProbs, 1, true);
		torch::Tensor qDist = torch::sum(curQ * curProbs, 1, true);
		torch::Tensor policyLoss = (-qDist - alpha * policyEntropy).mean();

		//		torch::Tensor policyLoss = (curProbs * (alpha * curLogProbs - curQ.detach())).sum(-1).mean();
		float policyLossValue = policyLoss.item<float>();

//		pOptimizer.zero_grad();
//		policyLoss.backward();
//		pOptimizer.step();


		//Alpha loss
//		torch::Tensor entropy;
//		{
//			torch::NoGradGuard guard;
//			auto probOutput = policyModel.forward(curStateTensor);
//			auto curProbs = torch::softmax(probOutput, -1);
//			auto curLogProbs = torch::log_softmax(probOutput, -1);
//			entropy = - (curProbs * curLogProbs).sum(-1);
//		}
//		torch::Tensor entropy = (curProbs * curLogProbs).sum(-1);
//		torch::Tensor alphaLoss = -(logAlpha * (entropy.detach() + dqnOption.targetEntropy)).mean();
//		torch::Tensor alphaLoss = (curProbs.detach() * (-logAlpha * curLogProbs.detach() - logAlpha * dqnOption.targetEntropy)).mean();
//		torch::Tensor entropy = (curProbs * ((-1) * curLogProbs)).sum(-1);
//		torch::Tensor alphaLoss = ((-logAlpha) * (dqnOption.targetEntropy - entropy.detach())).mean();
//		torch::Tensor alphaLoss = (logAlpha * (dqnOption.targetEntropy - entropy.detach())).mean();

//		torch::Tensor entropy = - (curProbs * curLogProbs).sum(-1);
//		torch::Tensor alphaLoss = (logAlpha * (entropy.detach() - dqnOption.targetEntropy)).mean();
		torch::Tensor alphaLoss = - torch::mean(logAlpha * (dqnOption.targetEntropy - policyEntropy.detach()));

//		LOG4CXX_INFO(logger, "entropy = " << entropy.mean());
//		LOG4CXX_INFO(logger, "curProbs = " << curProbs);
//		LOG4CXX_INFO(logger, "curLogProbs = " << curLogProbs);
//		LOG4CXX_INFO(logger, "entropy = " << entropy);
//		LOG4CXX_INFO(logger, "target = " << dqnOption.targetEntropy);
//		LOG4CXX_INFO(logger, "delta = " << (entropy.detach() - dqnOption.targetEntropy));
//		LOG4CXX_INFO(logger, "logAlpha = " << logAlpha);
//		LOG4CXX_INFO(logger, "loss = " << alphaLoss.item<float>());
		auto entropy = policyEntropy;
		float entropyValue = entropy.mean().abs().item<float>();
		float alphaLossValue = alphaLoss.item<float>();
		/*
		 * UPDATE
		 */
		qOptimizer1.zero_grad();
		qLoss1.backward();
		qOptimizer1.step();

		qOptimizer2.zero_grad();
		qLoss2.backward();
		qOptimizer2.step();

		pOptimizer.zero_grad();
		policyLoss.backward();
		pOptimizer.step();


		alphaOptimizer.zero_grad();
		alphaLoss.backward();
		alphaOptimizer.step();

		alpha = logAlpha.exp();
//		LOG4CXX_INFO(logger, "new alpha: " << alpha);

		if ((updateNum % dqnOption.logInterval) == 0) {
			lossStater.update({alpha.item<float>(), entropyValue, policyLossValue, alphaLossValue});
		}

		if ((updateNum % dqnOption.testGapEp) == 0) {
			if (dqnOption.toTest) {
				testModel(dqnOption.testEp, false);
			}
		}
	}

	save();
}

template <typename QNetType, typename PNetType, typename EnvType, typename PolicyType, typename QOptimizerType, typename POptimizerType, typename AlphaOptimizerType>
void Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::updateModel(bool force) {
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
void Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::testModel(const int epochNum, bool render) {
	std::vector<float> statRewards(dqnOption.envNum, 0);
	std::vector<float> statLens(dqnOption.envNum, 0);

	std::vector<float> stateVec = testEnv.reset();

	int testEp = 0;
	while (testEp < epochNum) {
		torch::Tensor inputTensor = torch::from_blob(stateVec.data(), inputShape).div(dqnOption.inputScale).to(deviceType);
		torch::Tensor outputTensor = policyModel.forward(inputTensor);
		torch::Tensor actionProbs = torch::softmax(outputTensor, -1);
		std::vector<int64_t> actions = policy.getTestActions(actionProbs);

		auto stepResult = testEnv.step(actions, render);
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
			testEp ++;
		}

		stateVec = nextInputVec;
	}
}


template <typename QNetType, typename PNetType, typename EnvType, typename PolicyType, typename QOptimizerType, typename POptimizerType, typename AlphaOptimizerType>
void Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::test(const int epochNum, bool render) {
	load();

	testModel(epochNum, render);
}

template <typename QNetType, typename PNetType, typename EnvType, typename PolicyType, typename QOptimizerType, typename POptimizerType, typename AlphaOptimizerType>
void Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::save() {
	if (!dqnOption.saveModel) {
		return;
	}

	std::string modelPathQ1 = dqnOption.savePathPrefix + "_q1_model.pt";
	torch::serialize::OutputArchive outputArchiveQ1;
	bModel1.save(outputArchiveQ1);
	outputArchiveQ1.save_to(modelPathQ1);
	LOG4CXX_INFO(logger, "Save model into " << modelPathQ1);

	std::string modelPathQ2 = dqnOption.savePathPrefix + "_q2_model.pt";
	torch::serialize::OutputArchive outputArchiveQ2;
	bModel2.save(outputArchiveQ2);
	outputArchiveQ2.save_to(modelPathQ2);
	LOG4CXX_INFO(logger, "Save model into " << modelPathQ2);

	std::string optPathQ1 = dqnOption.savePathPrefix + "_q1_optimizer.pt";
	torch::serialize::OutputArchive optimizerArchiveQ1;
	qOptimizer1.save(optimizerArchiveQ1);
	optimizerArchiveQ1.save_to(optPathQ1);
	LOG4CXX_INFO(logger, "Save optimizer into " << optPathQ1);

	std::string optPathQ2 = dqnOption.savePathPrefix + "_q2_optimizer.pt";
	torch::serialize::OutputArchive optimizerArchiveQ2;
	qOptimizer2.save(optimizerArchiveQ2);
	optimizerArchiveQ2.save_to(optPathQ2);
	LOG4CXX_INFO(logger, "Save optimizer into " << optPathQ2);

	std::string modelPathP = dqnOption.savePathPrefix + "_p_model.pt";
	torch::serialize::OutputArchive outputArchiveP;
	policyModel.save(outputArchiveP);
	outputArchiveP.save_to(modelPathP);
	LOG4CXX_INFO(logger, "Save model into " << modelPathP);

	std::string optPathP = dqnOption.savePathPrefix + "_p_optimizer.pt";
	torch::serialize::OutputArchive optimizerArchiveP;
	pOptimizer.save(optimizerArchiveP);
	optimizerArchiveP.save_to(optPathP);
	LOG4CXX_INFO(logger, "Save optimizer into " << optPathP);

	std::string alphaPath = dqnOption.savePathPrefix + "_alpha.pt";
	torch::save(logAlpha, alphaPath);
	LOG4CXX_INFO(logger, "Save alpha into " << alphaPath);

	std::string optPathA = dqnOption.savePathPrefix + "_a_optimizer.pt";
	torch::serialize::OutputArchive optimizerArchiveA;
	alphaOptimizer.save(optimizerArchiveA);
	optimizerArchiveA.save_to(optPathA);
	LOG4CXX_INFO(logger, "Save optimizer into " << optPathA);
}


template <typename QNetType, typename PNetType, typename EnvType, typename PolicyType, typename QOptimizerType, typename POptimizerType, typename AlphaOptimizerType>
void Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::load() {
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
	LOG4CXX_INFO(logger, "Load model from " << modelPathA);


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




#endif /* INC_ALG_SAC_HPP_ */
