/*
 * rnnappoupdater.hpp
 *
 *  Created on: Mar 5, 2022
 *      Author: zf
 */

#ifndef INC_ALG_RNN_APPO_RNNAPPOUPDATER_HPP_
#define INC_ALG_RNN_APPO_RNNAPPOUPDATER_HPP_




#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <tensorboard_logger.h>

#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>

#include "alg/utils/dqnoption.h"
#include "gymtest/env/envutils.h"
#include "gymtest/utils/stats.h"
//#include "gymtest/utils/lossstats.h"
#include "alg/rnn/appo/rnnappodataq.h"

template<typename NetType, typename OptimizerType>
class RnnAPPOUpdater {
private:
	NetType& bModel;
	OptimizerType& optimizer;
	AsyncRnnPPODataQ& q;

	const torch::Device deviceType;
	const at::IntArrayRef inputShape;

	const DqnOption dqnOption;

	uint32_t updateNum = 0;
	const int updateTargetGap; //TODO


	std::vector<int64_t> batchInputShape;
	std::vector<int64_t> trajInputShape;

	const int actionNum;


	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("testappornn");
	TensorBoardLogger tLogger;


	struct SeqSortMap {
		long seqLen;
		int index;
	};
public:
	RnnAPPOUpdater(NetType& behaviorModel, OptimizerType& iOptimizer,
			DqnOption option, AsyncRnnPPODataQ& iq, int actNum);
	~RnnAPPOUpdater() = default;
	RnnAPPOUpdater(const RnnAPPOUpdater& ) = delete;

	void train(const int updateNum);
//	void test(const int batchSize, const int epochNum);


	void load();
	void save();
};

template<typename NetType, typename OptimizerType>
RnnAPPOUpdater<NetType, OptimizerType>::RnnAPPOUpdater(NetType& behaviorModel,
		OptimizerType& iOptimizer,
		const DqnOption iOption,
		AsyncRnnPPODataQ& iq,
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
void RnnAPPOUpdater<NetType, OptimizerType>::train(const int updateNum) {
	LOG4CXX_INFO(logger, "training ");
	load();

	const int roundNum = dqnOption.trajStepNum / dqnOption.batchSize;
	int updateIndex = 0;

	while (updateIndex < updateNum) {
		tLogger.add_scalar("loss/q", updateIndex, (float)q.getStoreSize());


		std::vector<long> seqLens;
		std::vector<torch::Tensor> returnVec;
		std::vector<torch::Tensor> actionVec;
		std::vector<torch::Tensor> gaeVec;
		std::vector<torch::Tensor> oldPiVec;
		std::vector<torch::Tensor> inputStateVec;
		std::vector<std::vector<torch::Tensor>> hiddenStateVec;

		for (int i = 0; i < dqnOption.trajStepNum; i ++) {
			std::vector<torch::Tensor> datas = q.pop();

			//	std::vector<torch::Tensor> rc{stateTensor, actionTensor, returnTensor, gaeTensor, oldPiTensor};
			auto seqLen = datas[1].sizes()[0];
			seqLens.push_back(seqLen);
			inputStateVec.push_back(datas[0]);
			actionVec.push_back(datas[1]);
			returnVec.push_back(datas[2]);
			gaeVec.push_back(datas[3]);
			oldPiVec.push_back(datas[4]);

//			hiddenStateVec.push_back(std::vector<torch::Tensor>(datas.begin() + 5, datas.end()));
			std::vector<torch::Tensor> hState;
			for (int hIndex = 5; hIndex < datas.size(); hIndex ++) {
				hState.push_back(datas[hIndex]);
			}
			LOG4CXX_DEBUG(logger, "hState size " << hState.size());
			hiddenStateVec.push_back(hState);
		}
		LOG4CXX_DEBUG(logger, "hiddenStateVec " << hiddenStateVec.size());

		for (int epochIndex = 0; epochIndex < dqnOption.epochNum; epochIndex ++) {
			auto indiceTensor = torch::randperm(dqnOption.trajStepNum, longOpt);
			long* indicePtr = indiceTensor.data_ptr<long>();

			for (int roundIndex = 0; roundIndex < roundNum; roundIndex ++) {
				updateIndex ++;

				std::vector<SeqSortMap> seqMaps;
				for (int batchIndex = 0; batchIndex < dqnOption.batchSize; batchIndex ++) {
					int index = indicePtr[roundIndex * dqnOption.batchSize + batchIndex];
					seqMaps.push_back({seqLens[index], index});
				}
				std::sort(seqMaps.begin(), seqMaps.end(),
						[](const SeqSortMap& a, const SeqSortMap& b) -> bool {
					return a.seqLen > b.seqLen;
				});

				std::vector<torch::Tensor> returnPieceVec;
				std::vector<torch::Tensor> actionPieceVec;
				std::vector<torch::Tensor> gaePieceVec;
				std::vector<torch::Tensor> oldPiPieceVec;
				std::vector<torch::Tensor> inputStatePieceVec;
				std::vector<long> roundSeqLens(dqnOption.batchSize, 0);
				std::vector<std::vector<torch::Tensor>> hiddenStatePieceVec(dqnOption.gruCellNum);

				long sumLen = 0;
				for (int batchIndex = 0; batchIndex < dqnOption.batchSize; batchIndex ++) {
					int index = seqMaps[batchIndex].index;
					roundSeqLens[batchIndex] = seqLens[index];
					sumLen += seqLens[index];

					returnPieceVec.push_back(returnVec[index]);
					actionPieceVec.push_back(actionVec[index]);
					gaePieceVec.push_back(gaeVec[index]);
					oldPiPieceVec.push_back(oldPiVec[index]);
					inputStatePieceVec.push_back(inputStateVec[index]);

					for (int gruIndex = 0; gruIndex < dqnOption.gruCellNum; gruIndex ++) {
						hiddenStatePieceVec[gruIndex].push_back(hiddenStateVec[index][gruIndex]);
					}
				}

				torch::Tensor returnTensor = torch::cat(returnPieceVec, 0).view({sumLen, 1}).to(dqnOption.deviceType);
				torch::Tensor gaeTensor = torch::cat(gaePieceVec, 0).view({sumLen, 1}).to(dqnOption.deviceType);
				torch::Tensor oldPiTensor = torch::cat(oldPiPieceVec, 0).view({sumLen, 1}).to(dqnOption.deviceType);
				torch::Tensor actionTensor = torch::cat(actionPieceVec, 0).view({sumLen, 1}).to(dqnOption.deviceType);
				std::vector<long> seqShape{sumLen};
				seqShape.insert(seqShape.end(), dqnOption.inputShape.begin(), dqnOption.inputShape.end());
				torch::Tensor stateTensor = torch::cat(inputStatePieceVec, 0).view(seqShape);
				stateTensor = stateTensor.div(dqnOption.inputScale).to(dqnOption.deviceType);
				std::vector<torch::Tensor> hiddenState;
				for (int gruIndex = 0; gruIndex < dqnOption.gruCellNum; gruIndex ++) {
					hiddenState.push_back(torch::stack(hiddenStatePieceVec[gruIndex], 1));
				}
				LOG4CXX_DEBUG(logger, "returnTensor " << returnTensor.sizes());
				LOG4CXX_DEBUG(logger, "gaeTensor " << gaeTensor.sizes());
				LOG4CXX_DEBUG(logger, "oldPiTensor " << oldPiTensor.sizes());
				LOG4CXX_DEBUG(logger, "actionTensor " << actionTensor.sizes());

				auto output = bModel.forwardNext(stateTensor, dqnOption.batchSize, roundSeqLens, hiddenState, dqnOption.deviceType);
				torch::Tensor valueOutput = output[1];
				torch::Tensor actionOutput = output[0];

				torch::Tensor valueLossTensor = torch::nn::functional::mse_loss(valueOutput, returnTensor);

				torch::Tensor actionPiTensor = torch::softmax(actionOutput, -1);
				torch::Tensor actionPi = actionPiTensor.gather(-1, actionTensor);
				torch::Tensor ratio = actionPi / oldPiTensor;
//				LOG4CXX_INFO(logger, "ratio is " << ratio);
				float kl = ratio.mean().to(torch::kCPU).item<float>();
				if ((kl > (1 + dqnOption.maxKl)) || (kl < (1 - dqnOption.maxKl))) {
					epochIndex = dqnOption.epochNum;
					break;
				}

				auto advTensor = gaeTensor;
				auto sur0 = ratio * advTensor.detach();
				auto sur1 = torch::clamp(ratio, 1 - dqnOption.ppoEpsilon, 1 + dqnOption.ppoEpsilon) * advTensor.detach();
				LOG4CXX_DEBUG(logger, "sur0 = " << sur0.sizes());
				LOG4CXX_DEBUG(logger, "sur1 = " << sur1.sizes());
				torch::Tensor actLossTensor = torch::min(sur0, sur1).mean() * (-1);
				LOG4CXX_DEBUG(logger, "actLossTensor = " << actLossTensor);


				torch::Tensor actionLogDistTensor = torch::log_softmax(actionOutput, -1);
				LOG4CXX_DEBUG(logger, "actionLogTensor " << actionLogDistTensor.sizes());
				torch::Tensor entropyTensor = (-1) * (actionLogDistTensor * actionPiTensor).sum(-1).mean();
				LOG4CXX_DEBUG(logger, "entropy = " << entropyTensor);

				torch::Tensor lossTensor = actLossTensor + dqnOption.valueCoef * valueLossTensor - dqnOption.entropyCoef * entropyTensor;

				optimizer.zero_grad();
				lossTensor.backward();
				torch::nn::utils::clip_grad_norm_(bModel.parameters(), dqnOption.maxGradNormClip);
				optimizer.step();

				if ((updateNum % dqnOption.logInterval) == 0) {
					//print and log
					float lossV = lossTensor.item<float>();
					float vLossV = valueLossTensor.item<float>();
					float aLossV = actLossTensor.item<float>();
					float entropyV = entropyTensor.item<float>();
					float valueV = valueOutput.mean().item<float>();
					float seqMeanV = (float)sumLen / roundSeqLens.size();
					LOG4CXX_DEBUG(logger, "loss" << updateIndex << "-" << epochIndex << "-" << roundIndex << ": " << lossV
							<< ", " << vLossV << ", " << aLossV << ", " << entropyV << ", " << kl);

					tLogger.add_scalar("loss/loss", updateIndex, lossV);
					tLogger.add_scalar("loss/vLoss", updateIndex, vLossV);
					tLogger.add_scalar("loss/aLoss", updateIndex, aLossV);
					tLogger.add_scalar("loss/entropy", updateIndex, entropyV);
					tLogger.add_scalar("loss/v", updateIndex, valueV);
					tLogger.add_scalar("loss/kl", updateIndex, kl);
					tLogger.add_scalar("loss/seq", updateIndex, seqMeanV);
//					tLogger.add_scalar("loss/update", updateIndex, (float)updateNum);
				}
			}

		}

	}

	save();
}


template<typename NetType, typename OptimizerType>
void RnnAPPOUpdater<NetType, OptimizerType>::save() {
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
void RnnAPPOUpdater<NetType, OptimizerType>::load() {
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


#endif /* INC_ALG_RNN_APPO_RNNAPPOUPDATER_HPP_ */
