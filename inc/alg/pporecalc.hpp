/*
 * pporecalc.hpp
 *
 *  Created on: Aug 15, 2021
 *      Author: zf
 */

#ifndef INC_ALG_PPORECALC_HPP_
#define INC_ALG_PPORECALC_HPP_



#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/stats.h"
#include "utils/dqnoption.h"

#include "envstep.hpp"

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
class PPORecalc: private A2CStoreEnvStep<NetType, EnvType, PolicyType> {
private:
	NetType& bModel;
	EnvType& env;
	EnvType& testEnv;
	PolicyType& policy;
	OptimizerType& optimizer;
	const torch::Device deviceType;
	const at::IntArrayRef inputShape;

	const DqnOption dqnOption;

	std::vector<int64_t> batchInputShape;
	std::vector<int64_t> trajInputShape;

	const int actionNum;

	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("pporecalc");

	AlgTester<NetType, EnvType, PolicyType> tester;
	using A2CStoreEnvStep<NetType, EnvType, PolicyType>::tLogger;

	std::vector<torch::Tensor> calcGae(torch::Tensor valueTensor, torch::Tensor rewardTensor, torch::Tensor doneTensor, torch::Tensor lastValueTensor);

public:
	PPORecalc(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, DqnOption option, int actNum);
	~PPORecalc() = default;
	PPORecalc(const PPORecalc& ) = delete;

	void train(const int updateNum);
	void test();


	void load();
	void save();
};

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
PPORecalc<NetType, EnvType, PolicyType, OptimizerType>::PPORecalc(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy,
		OptimizerType& iOptimizer,
		const DqnOption iOption,
		int actNum):
	A2CStoreEnvStep<NetType, EnvType, PolicyType>(iOption),
	bModel(behaviorModel),
	env(iEnv),
	testEnv(tEnv),
	policy(iPolicy),
	optimizer(iOptimizer),
	dqnOption(iOption),
	deviceType(iOption.deviceType),
	inputShape(iOption.inputShape),
	actionNum(actNum),
	tester(behaviorModel, tEnv, iPolicy, iOption, tLogger)
{
		batchInputShape.push_back(dqnOption.envNum * dqnOption.batchSize);
		trajInputShape.push_back(dqnOption.trajStepNum * dqnOption.envNum);
		for (int i = 1; i < inputShape.size(); i ++) {
			batchInputShape.push_back(inputShape[i]);
			trajInputShape.push_back(inputShape[i]);
		}
}


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
std::vector<torch::Tensor> PPORecalc<NetType, EnvType, PolicyType, OptimizerType>::calcGae(
		torch::Tensor valueTensor, torch::Tensor rewardTensor, torch::Tensor doneTensor, torch::Tensor lastValueTensor) {
	at::IntArrayRef batchValueShape{dqnOption.trajStepNum, dqnOption.envNum, 1};

	torch::Tensor gaeReturns = torch::zeros(batchValueShape);
	torch::Tensor returns = torch::zeros(batchValueShape);
	torch::Tensor nextValueTensor = lastValueTensor.to(torch::kCPU).detach();
	torch::Tensor gaeReturn = torch::zeros({dqnOption.envNum, 1});

	for (int i = dqnOption.trajStepNum - 1; i >= 0; i --) {
		torch::Tensor delta = rewardTensor[i] + dqnOption.gamma * nextValueTensor * doneTensor[i] - valueTensor[i];
		gaeReturn = delta + dqnOption.ppoLambda * dqnOption.gamma * gaeReturn * doneTensor[i];
		gaeReturns[i].copy_(gaeReturn);

		nextValueTensor = valueTensor[i];
	}
	returns = gaeReturns + valueTensor; //from baseline3.

	gaeReturns = gaeReturns.detach().view({dqnOption.trajStepNum * dqnOption.envNum, 1}).to(deviceType);
	returns = returns.detach().view({dqnOption.trajStepNum * dqnOption.envNum, 1}).to(deviceType);

	return {gaeReturns, returns};
}


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PPORecalc<NetType, EnvType, PolicyType, OptimizerType>::train(const int updateNum) {
	LOG4CXX_INFO(logger, "training ");
	load();

	int updateIndex = 0;
	int stepNum = 0;


	this->stateVec = env.reset();
	while (stepNum < updateNum) {
		LOG4CXX_DEBUG(logger, "---------------------------------------> update  " << updateIndex);
		this->steps(bModel, env, policy, dqnOption.trajStepNum, stepNum);
		LOG4CXX_DEBUG(logger, "Collect " << dqnOption.trajStepNum << " step samples ");


		//Calculate GAE return
		torch::Tensor lastStateTensor = torch::from_blob(this->stateVec.data(), inputShape).div(dqnOption.inputScale).to(deviceType);
//		auto lastRc = bModel.forward(lastStateTensor);
//		torch::Tensor lastValueTensor = lastRc[1];
//		LOG4CXX_DEBUG(logger, "Get last value " << lastValueTensor.sizes());

		//TODO: Put all tensors in CPU to save storage for inputState tensors?
		at::IntArrayRef batchValueShape{dqnOption.trajStepNum, dqnOption.envNum, 1};

		auto doneData = EnvUtils::FlattenVector(this->donesVec);
		auto rewardData = EnvUtils::FlattenVector(this->rewardsVec);
		torch::Tensor rewardTensor = torch::from_blob(rewardData.data(), batchValueShape).div(dqnOption.rewardScale).clamp(dqnOption.rewardMin, dqnOption.rewardMax);
		torch::Tensor doneTensor = torch::from_blob(doneData.data(), batchValueShape);

		//Calculate old log Pi
		torch::Tensor oldDistTensor = torch::stack(this->pisVec, 0).view({dqnOption.trajStepNum, dqnOption.envNum, actionNum});
		oldDistTensor = torch::softmax(oldDistTensor, -1).to(deviceType);
		LOG4CXX_DEBUG(logger, "oldDistTensor: " << oldDistTensor.sizes());

		auto actionData = EnvUtils::FlattenVector(this->actionsVec);
		torch::Tensor oldActionTensor = torch::from_blob(actionData.data(), {dqnOption.trajStepNum, dqnOption.envNum, 1}, longOpt).to(deviceType);
		LOG4CXX_DEBUG(logger, "oldActionTensor: " << oldActionTensor.sizes());
		torch::Tensor oldPiTensor = oldDistTensor.gather(-1, oldActionTensor).detach();
		LOG4CXX_DEBUG(logger, "oldPiTensor: " << oldPiTensor.sizes());

		const int roundNum = dqnOption.trajStepNum / dqnOption.batchSize;

		auto stateData = EnvUtils::FlattenVector(this->statesVec);
		auto stateTensor = torch::from_blob(stateData.data(), trajInputShape).div(dqnOption.inputScale).to(deviceType);
		LOG4CXX_DEBUG(logger, "stateTensor: " << stateTensor.sizes());

		oldPiTensor = oldPiTensor.view({dqnOption.trajStepNum * dqnOption.envNum, 1});
		oldActionTensor = oldActionTensor.view({dqnOption.trajStepNum * dqnOption.envNum, 1});


//		LOG4CXX_INFO(logger, "shuffled indice " << indice);
//		LOG4CXX_INFO(logger, "indiceTensor \n" << indiceTensor);
		auto pieceLen = dqnOption.batchSize * dqnOption.envNum;

//		auto indiceTensor = torch::randperm(dqnOption.trajStepNum * dqnOption.envNum, longOpt).view({-1, dqnOption.batchSize * dqnOption.envNum}).to(deviceType);
		for (int epochIndex = 0; epochIndex < dqnOption.epochNum; epochIndex ++) {
			//Recalculation GAE per epoch
			std::vector<torch::Tensor> gaeRc;
			//TODO: should rc calculated per round or per epoch?
			{
				torch::Tensor valueTensor;
				torch::NoGradGuard guard;
				auto rc = bModel.forward(stateTensor);
				valueTensor = rc[1];
				valueTensor = valueTensor.view(batchValueShape).to(torch::kCPU);

				auto lastRc = bModel.forward(lastStateTensor);
				auto lastValueTensor = lastRc[1];
				gaeRc = calcGae(valueTensor, rewardTensor, doneTensor, lastValueTensor);
			}
			torch::Tensor gaeReturns = gaeRc[0];
			torch::Tensor returns = gaeRc[1];

			auto indiceTensor = torch::randperm(dqnOption.trajStepNum * dqnOption.envNum, longOpt).view({-1, dqnOption.batchSize * dqnOption.envNum}).to(deviceType);

			for (int roundIndex = 0; roundIndex < roundNum; roundIndex ++) {
				updateIndex ++;

				//fetch data
//				torch::Tensor indexPiece = indiceTensor.narrow(0, pieceLen * roundIndex, pieceLen);
				auto indexPiece = indiceTensor[roundIndex];
//				LOG4CXX_INFO(logger, "index piece: " << indexPiece);
//				LOG4CXX_INFO(logger, "state size: " << stateTensor.sizes());
				torch::Tensor stateInput = stateTensor.index_select(0, indexPiece);
//				LOG4CXX_INFO(logger, "state input sizes: " << stateInput.sizes());
				torch::Tensor returnPiece = returns.index_select(0, indexPiece);
				torch::Tensor gaePiece = gaeReturns.index_select(0, indexPiece);
				torch::Tensor oldPiPiece = oldPiTensor.index_select(0, indexPiece);
				torch::Tensor oldActionPiece = oldActionTensor.index_select(0, indexPiece);
//				torch::Tensor valuePiece = valueTensor.index_select(0, indexPiece);
//				torch::Tensor valueOutput = valueTensor.index_select(0, indexPiece);
//				torch::Tensor actionOutput = actionOutputTensor.idnex_select(0, indexPiece);

				auto rc = bModel.forward(stateInput);
				torch::Tensor valueOutput = rc[1];
				torch::Tensor actionOutput = rc[0];


				torch::Tensor valueLossTensor = torch::nn::functional::mse_loss(valueOutput, returnPiece);

				//action loss
				//actionPi and oldPi are logPi
				torch::Tensor advTensor = gaePiece;
				if (dqnOption.normReward) {
					advTensor = (advTensor - advTensor.mean()) / (advTensor.std() + 1e-7);
				}
				torch::Tensor actionPiTensor = torch::softmax(actionOutput, -1);
				torch::Tensor actionPi = actionPiTensor.gather(-1, oldActionPiece);
				auto ratio = actionPi / oldPiPiece;
//				LOG4CXX_INFO(logger, "ratio is " << ratio);
				float kl = ratio.mean().to(torch::kCPU).item<float>();


				auto sur0 = ratio * advTensor.detach();
				auto sur1 = torch::clamp(ratio, 1 - dqnOption.ppoEpsilon, 1 + dqnOption.ppoEpsilon) * advTensor.detach();
				LOG4CXX_DEBUG(logger, "sur1 = " << sur1);
				torch::Tensor actLossTensor = torch::min(sur0, sur1).mean() * (-1);
				LOG4CXX_DEBUG(logger, "actLossTensor = " << actLossTensor);

				//entropy loss
				torch::Tensor actionLogDistTensor = torch::log_softmax(actionOutput, -1);
				LOG4CXX_DEBUG(logger, "actionLogTensor " << actionLogDistTensor);
				torch::Tensor entropyTensor = (-1) * (actionLogDistTensor * actionPiTensor).sum(-1).mean();
				LOG4CXX_DEBUG(logger, "entropy = " << entropyTensor);

				//overall loss
				torch::Tensor lossTensor = actLossTensor + dqnOption.valueCoef * valueLossTensor - dqnOption.entropyCoef * entropyTensor;

				optimizer.zero_grad();
				lossTensor.backward();
				torch::nn::utils::clip_grad_norm_(bModel.parameters(), dqnOption.maxGradNormClip);
				optimizer.step();

				//print and log
				if ((updateIndex % dqnOption.logInterval) == 0) {
					//print and log
					float lossV = lossTensor.item<float>();
					float vLossV = valueLossTensor.item<float>();
					float aLossV = actLossTensor.item<float>();
					float entropyV = entropyTensor.item<float>();
					float valueV = valueOutput.mean().item<float>();
					LOG4CXX_DEBUG(logger, "loss" << updateIndex << "-" << epochIndex << "-" << roundIndex << ": " << lossV
							<< ", " << vLossV << ", " << aLossV << ", " << entropyV << ", " << kl);

					tLogger.add_scalar("loss/loss", updateIndex, lossV);
					tLogger.add_scalar("loss/vLoss", updateIndex, vLossV);
					tLogger.add_scalar("loss/aLoss", updateIndex, aLossV);
					tLogger.add_scalar("loss/entropy", updateIndex, entropyV);
					tLogger.add_scalar("loss/v", updateIndex, valueV);
					tLogger.add_scalar("loss/kl", updateIndex, kl);
				}
			}
		}

		if ((stepNum % dqnOption.testGapEp) == 0) {
			test();
		}
	}

	save();
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PPORecalc<NetType, EnvType, PolicyType, OptimizerType>::test() {
	LOG4CXX_INFO(logger, "To test " << dqnOption.testEp << " episodes");
	if (!dqnOption.toTest) {
		return;
	}

	tester.testAC();
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PPORecalc<NetType, EnvType, PolicyType, OptimizerType>::save() {
	if (!dqnOption.saveModel) {
		return;
	}

	AlgUtils::SaveModel(bModel, optimizer, dqnOption.savePathPrefix, logger);
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void PPORecalc<NetType, EnvType, PolicyType, OptimizerType>::load() {
	if (!dqnOption.loadModel) {
		return;
	}

	AlgUtils::LoadModel(bModel, optimizer, dqnOption.loadOptimizer, dqnOption.loadPathPrefix, logger);
}


#endif /* INC_ALG_PPOSHAREDTEST_HPP_ */

