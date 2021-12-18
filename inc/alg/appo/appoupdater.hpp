/*
 * appoupdater.hpp
 *
 *  Created on: Dec 15, 2021
 *      Author: zf
 */

#ifndef INC_ALG_APPO_APPOUPDATER_HPP_
#define INC_ALG_APPO_APPOUPDATER_HPP_


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
#include "alg/dqnoption.h"
#include "appodataq.h"

template<typename NetType, typename OptimizerType>
class APPOUpdater {
private:
	NetType& bModel;
	OptimizerType& optimizer;
	AsyncPPODataQ& q;

	const torch::Device deviceType;
	const at::IntArrayRef inputShape;

	const DqnOption dqnOption;

	uint32_t updateNum = 0;
	const int updateTargetGap; //TODO


	std::vector<int64_t> batchInputShape;
	std::vector<int64_t> trajInputShape;

	const int actionNum;


	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("apposhared");
	TensorBoardLogger tLogger;


public:
	APPOUpdater(NetType& behaviorModel, OptimizerType& iOptimizer,
			DqnOption option, AsyncPPODataQ& iq, int actNum);
	~APPOUpdater() = default;
	APPOUpdater(const APPOUpdater& ) = delete;

	void train(const int updateNum);
//	void test(const int batchSize, const int epochNum);


	void load();
	void save();
};

template<typename NetType, typename OptimizerType>
APPOUpdater<NetType, OptimizerType>::APPOUpdater(NetType& behaviorModel,
		OptimizerType& iOptimizer,
		const DqnOption iOption,
		AsyncPPODataQ& iq,
		int actNum):
	bModel(behaviorModel),
	optimizer(iOptimizer),
	dqnOption(iOption),
	q(iq),
	deviceType(iOption.deviceType),
	inputShape(iOption.inputShape),
	tLogger(iOption.tensorboardLogPath.c_str()),
	updateTargetGap(iOption.targetUpdate),
	actionNum(actNum)
{
	assert((dqnOption.trajStepNum % dqnOption.appoRoundNum) == 0);


	batchInputShape.push_back(dqnOption.trajStepNum);
	trajInputShape.push_back(dqnOption.trajStepNum);
	for (int i = 1; i < inputShape.size(); i ++) {
		batchInputShape.push_back(inputShape[i]);
		trajInputShape.push_back(inputShape[i]);
	}


//	LOG4CXX_INFO(logger, "indice after initiation \n " << indice);
	std::srand(unsigned (std::time(0)));
}


template<typename NetType, typename OptimizerType>
void APPOUpdater<NetType, OptimizerType>::train(const int updateNum) {
	LOG4CXX_INFO(logger, "training ");
	load();

	int updateIndex = 0;
	uint64_t trainStepIndex = 0;
	const int roundItemNum = dqnOption.trajStepNum / dqnOption.appoRoundNum;

	/*
	 * 	auto stateTensor = torch::cat(states, 0);
	auto returnTensor = torch::cat(returns, 0);
	auto gaeTensor = torch::cat(returns, 0);
	auto oldPiTensor = torch::cat(oldPis, 0);
	auto actionTensor = torch::cat(actions, 0);
	 */
	//trajStepNum for all
	while (updateIndex < updateNum) {
		updateIndex ++;

		const auto storeSize = q.getStoreSize();
		tLogger.add_scalar("updater/traffic", updateIndex, (float)storeSize);

		std::vector<torch::Tensor> datas = q.pop(dqnOption.trajStepNum);
		auto stateTensor = datas[0].to(deviceType);
		auto returnTensor = datas[1].to(deviceType);
		auto gaeTensor = datas[2].to(deviceType);
		auto oldPiTensor = datas[3].to(deviceType);
		auto oldActionTensor = datas[4].to(deviceType);

		float gaeMean = gaeTensor.mean().item<float>();
		tLogger.add_scalar("updater/gae", updateIndex,  gaeMean);
//		LOG4CXX_INFO(logger, "data size " << datas[0].sizes());
//		LOG4CXX_INFO(logger, "return size " << datas[1].sizes());
//		LOG4CXX_INFO(logger, "gae size " << datas[2].sizes());
//		LOG4CXX_INFO(logger, "pi size " << datas[3].sizes());
//		LOG4CXX_INFO(logger, "action size " << datas[4].sizes());
		bool moreEpoch = true;
		for (int epochIndex = 0; (epochIndex < dqnOption.epochNum) && moreEpoch; epochIndex ++) {
			auto indiceTensor = torch::randperm(dqnOption.trajStepNum, longOpt).view({-1, roundItemNum}).to(deviceType);
//			LOG4CXX_INFO(logger, "indiceTensor: " << indiceTensor.sizes());

			for (int roundIndex = 0; roundIndex < dqnOption.appoRoundNum; roundIndex ++) {
				trainStepIndex ++;

				auto indexPiece = indiceTensor[roundIndex];
//				LOG4CXX_INFO(logger, "indexPiece: \n" << indexPiece);

//				auto stateInput = datas[0].index_select(0, indexPiece).to(deviceType);
//				auto returnPiece = datas[1].index_select(0, indexPiece).to(deviceType);
//				auto gaePiece = datas[2].index_select(0, indexPiece).to(deviceType);
//				auto oldPiPiece = datas[3].index_select(0, indexPiece).to(deviceType);
//				auto oldActionPiece = datas[4].index_select(0, indexPiece).to(deviceType);
				auto stateInput = stateTensor.index_select(0, indexPiece);
				auto returnPiece = returnTensor.index_select(0, indexPiece);
				auto gaePiece = gaeTensor.index_select(0, indexPiece);
				auto oldPiPiece = oldPiTensor.index_select(0, indexPiece);
				auto oldActionPiece = oldActionTensor.index_select(0, indexPiece);
//				LOG4CXX_INFO(logger, "updater gae \n" << gaePiece);

				auto rc = bModel.forward(stateInput);
				auto valueOutput = rc[1];
				auto actionOutput = rc[0];

				torch::Tensor valueLossTensor = torch::nn::functional::mse_loss(valueOutput, returnPiece.detach());

				auto actionPiTensor = torch::softmax(actionOutput, -1);
				auto actionPi = actionPiTensor.gather(-1, oldActionPiece);
				torch::Tensor ratio = actionPi / oldPiPiece;

				auto sur0 = ratio * gaePiece.detach();
				auto sur1 = torch::clamp(ratio, 1 - dqnOption.ppoEpsilon, 1 + dqnOption.ppoEpsilon) * gaePiece.detach();
				torch::Tensor actLossTensor = torch::min(sur0, sur1).mean() * (-1);

				auto actionLogPi = torch::log_softmax(actionOutput, -1);
				torch::Tensor entropyTensor = (-1) * (actionLogPi * actionPiTensor).sum(-1).mean();

				torch::Tensor loss = actLossTensor + dqnOption.valueCoef * valueLossTensor - dqnOption.entropyCoef * entropyTensor;

				if ((updateIndex % dqnOption.logInterval) == 0) {
					torch::NoGradGuard guard;

					auto lossV = loss.item<float>();
					auto vLossV = valueLossTensor.item<float>();
					auto aLossV = actLossTensor.item<float>();
					auto eLossV = entropyTensor.item<float>();
					auto kl = ratio.mean().to(torch::kCPU).item<float>();
					auto gaeValue = gaePiece.detach().mean().item<float>();
					/*
					 *                 with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
					 */

					tLogger.add_scalar("train/loss", trainStepIndex, lossV);
					tLogger.add_scalar("train/vLoss", trainStepIndex, vLossV);
					tLogger.add_scalar("train/aLoss", trainStepIndex, aLossV);
					tLogger.add_scalar("train/entropy", trainStepIndex, eLossV);
					tLogger.add_scalar("train/kl", trainStepIndex, kl);
					tLogger.add_scalar("train/gae", trainStepIndex, gaeValue);

					if ((kl > (1 + dqnOption.ppoEpsilon)) || (kl < (1 - dqnOption.ppoEpsilon))) {
						moreEpoch = false;
					}
				}

				optimizer.zero_grad();
				loss.backward();
				torch::nn::utils::clip_grad_norm_(bModel.parameters(), dqnOption.maxGradNormClip);
				optimizer.step();
			}
		}
	}
	save();
}

//template<typename NetType, typename EnvType, typename PolicyType, typename OptimizerType>
//void APPOUpdater<NetType, EnvType, PolicyType, OptimizerType>::test(const int batchSize, const int epochNum) {
//	LOG4CXX_INFO(logger, "To test " << epochNum << " episodes");
//	if (!dqnOption.toTest) {
//		return;
//	}
//
//	int epCount = 0;
//	std::vector<float> statRewards(batchSize, 0);
//	std::vector<float> statLens(batchSize, 0);
//
//	torch::NoGradGuard guard;
//	std::vector<float> states = testEnv.reset();
//	while (epCount < epochNum) {
////		torch::Tensor stateTensor = torch::from_blob(states.data(), inputShape).to(deviceType);
//		torch::Tensor stateTensor = torch::from_blob(states.data(), inputShape).div(dqnOption.inputScale).to(deviceType);
//
//		std::vector<torch::Tensor> rc = bModel.forward(stateTensor);
//		auto actionOutput = rc[0]; //TODO: detach?
//		auto valueOutput = rc[1];
//		auto actionProbs = torch::softmax(actionOutput, -1);
//		//TODO: To replace by getActions
//		std::vector<int64_t> actions = policy.getTestActions(actionProbs);
//
//		auto stepResult = testEnv.step(actions, true);
//		auto nextStateVec = std::get<0>(stepResult);
//		auto rewardVec = std::get<1>(stepResult);
//		auto doneVec = std::get<2>(stepResult);
//
//		Stats::UpdateReward(statRewards, rewardVec);
//		Stats::UpdateLen(statLens);
//
//		for (int i = 0; i < batchSize; i ++) {
//			if (doneVec[i]) {
//				LOG4CXX_DEBUG(logger, "testEnv " << i << "done");
////				auto resetResult = env.reset(i);
//				//udpate nextstatevec, target mask
////				std::copy(resetResult.begin(), resetResult.end(), nextStateVec.begin() + (offset * i));
//				epCount ++;
//
//				testStater.update(statLens[i], statRewards[i]);
//				statLens[i] = 0;
//				statRewards[i] = 0;
////				stater.printCurStat();
//				LOG4CXX_INFO(logger, "test -----------> " << testStater);
//
//			}
//		}
//		states = nextStateVec;
//	}
//}

template<typename NetType, typename OptimizerType>
void APPOUpdater<NetType, OptimizerType>::save() {
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

template<typename NetType, typename OptimizerType>
void APPOUpdater<NetType, OptimizerType>::load() {
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


#endif /* INC_ALG_APPO_APPOUPDATER_HPP_ */
