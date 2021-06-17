/*
 * a2c.hpp
 *
 *  Created on: Apr 29, 2021
 *      Author: zf
 */

#ifndef INC_ALG_A2C_HPP_
#define INC_ALG_A2C_HPP_



#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <vector>

#include "gymtest/env/envutils.h"
#include "gymtest/utils/replaybuffer.h"
#include "gymtest/utils/stats.h"
#include "dqnoption.h"
#include "gymtest/utils/lossstats.h"

//A2C 1 step Q
template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
class A2C {
private:
	NetType& bModel;
//	NetType& tModel;
	EnvType& env;
	EnvType& testEnv;
	PolicyType& policy;
	OptimizerType& optimizer;
	const torch::Device deviceType;
	const at::IntArrayRef inputShape;

	const DqnOption dqnOption;
	Stats stater;
	Stats testStater;
	LossStats lossStater;


	const int batchSize;
	int offset = 0;

	uint32_t updateNum = 0;
	const int updateTargetGap; //TODO

	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("a2c1stepq");
	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);

	void test();

	void save();
	void load();
public:
	const float gamma;
	const int testEp = 16;

	A2C(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer, DqnOption option);
	~A2C() = default;
	A2C(const A2C& ) = delete;

	void train(const int epNum);
};

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
A2C<NetType, EnvType, PolicyType, OptimizerType>::A2C(NetType& behaviorModel, EnvType& iEnv, EnvType& tEnv, PolicyType& iPolicy, OptimizerType& iOptimizer,
		const DqnOption iOption):
	bModel(behaviorModel),
//	tModel(trainModel),
	env(iEnv),
	testEnv(tEnv),
	policy(iPolicy),
	optimizer(iOptimizer),
	dqnOption(iOption),
	deviceType(iOption.deviceType),
	inputShape(iOption.inputShape),
	gamma(iOption.gamma),
	stater(iOption.statPathPrefix + "_stat.txt"),
	testStater(iOption.statPathPrefix + "_test.txt"),
	lossStater(iOption.statPathPrefix + "_loss.txt"),
	batchSize(iOption.batchSize),
	updateTargetGap(iOption.targetUpdate) {
	offset = 1;
	for (int i = 1; i < inputShape.size(); i ++) {
		offset *= inputShape[i];
	}
}


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2C<NetType, EnvType, PolicyType, OptimizerType>::train(const int epNum) {
	load();

	int step = 0;
	int epCount = 0;

	std::vector<float> statRewards(batchSize, 0);
	std::vector<float> statLens(batchSize, 0);

	std::vector<float> states = env.reset();
	while (epCount < epNum) {
		step ++;

		torch::Tensor stateTensor = torch::from_blob(states.data(), inputShape).to(deviceType);//.div(255);
		std::vector<torch::Tensor> rc = bModel.forward(stateTensor);
		auto actionOutput = rc[0]; //TODO: detach?
		auto valueOutput = rc[1];
		auto actionProbs = torch::softmax(actionOutput, -1);
		std::vector<int64_t> actions = policy.getActions(actionProbs);

		auto stepResult = env.step(actions, false);
		auto nextStateVec = std::get<0>(stepResult);
		auto rewardVec = std::get<1>(stepResult);
		auto doneVec = std::get<2>(stepResult);

		Stats::UpdateReward(statRewards, rewardVec);
		Stats::UpdateLen(statLens);

		torch::Tensor doneTensor = torch::ones({batchSize, 1}).to(deviceType);
		for (int i = 0; i < batchSize; i ++) {
			if (doneVec[i]) {
				LOG4CXX_DEBUG(logger, "env " << i << "done");
				doneTensor[i] = 0;

//				auto resetResult = env.reset(i);
				//udpate nextstatevec, target mask
//				std::copy(resetResult.begin(), resetResult.end(), nextStateVec.begin() + (offset * i));
				epCount ++;

				stater.update(statLens[i], statRewards[i]);
				statLens[i] = 0;
				statRewards[i] = 0;
//				stater.printCurStat();
				LOG4CXX_INFO(logger, stater);

//				 if (epCount < dqnOption.exploreStep) {
//					 float exploreRate = dqnOption.exploreBegin - (dqnOption.exploreBegin - dqnOption.exploreEnd) * (epCount / dqnOption.exploreStep);
//					 policy.updateEpsilon(exploreRate);
//				 }
//			    if (epCount % dqnOption.testGapEp == 0) {
//			    	test();
//			    }
			}
		}

		torch::Tensor rewardTensor = torch::from_blob(rewardVec.data(), {batchSize, 1}).to(deviceType);
		torch::Tensor actionTensor = torch::from_blob(actions.data(), {batchSize, 1}, longOpt).to(deviceType);
		torch::Tensor nextStateTensor = torch::from_blob(nextStateVec.data(), inputShape).to(deviceType);//.div(255);

		torch::Tensor nextVTensor;
		{
			torch::NoGradGuard();
			nextVTensor = bModel.forward(nextStateTensor)[1].detach();
		}
		torch::Tensor returnTensor = nextVTensor * doneTensor * gamma + rewardTensor;
		torch::Tensor advTensor = returnTensor - valueOutput;
		torch::Tensor valueLoss = 0.5 * advTensor.pow(2).mean();

		torch::Tensor actionLogProbs = torch::log_softmax(actionOutput, -1);
//		torch::Tensor actionProbs = torch::softmax(actionOutput, -1);
//		actionProbs = actionProbs.clamp(1.21e-7, 1.0f - 1.21e-7);
		torch::Tensor entropy = - (actionLogProbs * actionProbs).sum(-1).mean();

		torch::Tensor actPi = actionLogProbs.gather(-1, actionTensor); //TODO: actionTensor detach?
		torch::Tensor actionLoss = - (actPi * advTensor.detach()).mean();

		torch::Tensor loss = valueLoss + actionLoss - (1e-3) * entropy;
		LOG4CXX_INFO(logger, "------------> Loss: " << loss.item<float>() << ", " << valueLoss.item<float>() << ", " << actionLoss.item<float>());
		lossStater.update({loss.item<float>(), valueLoss.item<float>(), actionLoss.item<float>(), entropy.item<float>()});

    	optimizer.zero_grad();
	    loss.backward();
		torch::nn::utils::clip_grad_value_(bModel.parameters(), 1);
	    optimizer.step();

//	    if (step < dqnOption.exploreStep) {
//	        float exploreRate = dqnOption.exploreBegin - (dqnOption.exploreBegin - dqnOption.exploreEnd) * (step / dqnOption.exploreStep);
//	        policy.updateEpsilon(exploreRate);
//	    }

	    states = nextStateVec;
	}

	save();
}

template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2C<NetType, EnvType, PolicyType, OptimizerType>::test() {
	if (!dqnOption.toTest) {
		return;
	}

	bModel.eval();

	int epCount = 0;

	std::vector<float> statRewards(batchSize, 0);
	std::vector<float> statLens(batchSize, 0);

//	model.eval();
	std::vector<float> stateVec = testEnv.reset();
	while (epCount < testEp) {
		torch::Tensor inputTensor = torch::from_blob(stateVec.data(), inputShape).to(deviceType);
		torch::Tensor outputTensor = bModel.forward(inputTensor)[0];

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

		for (int i = 0; i < batchSize; i ++) {
			if (doneVec[i]) {
				LOG4CXX_DEBUG(logger, "ep " << epCount << "done");
				auto resetResult = testEnv.reset(i);
				std::copy(resetResult.begin(), resetResult.end(), nextStateVec.begin() + (offset * i));
				epCount ++;

				testStater.update(statLens[0], statRewards[0]);
				statLens[0] = 0;
				statRewards[0] = 0;
				LOG4CXX_INFO(logger, "test " << testStater);
			}
		}
    	stateVec = nextStateVec;
	}

	bModel.train();
}


template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
void A2C<NetType, EnvType, PolicyType, OptimizerType>::save() {
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
void A2C<NetType, EnvType, PolicyType, OptimizerType>::load() {
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

#endif /* INC_ALG_A2C_HPP_ */
