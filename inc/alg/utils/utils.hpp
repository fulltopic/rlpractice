/*
 * utils.hpp
 *
 *  Created on: Dec 10, 2021
 *      Author: zf
 */

#ifndef INC_ALG_UTILS_HPP_
#define INC_ALG_UTILS_HPP_

#include <torch/torch.h>

#include <log4cxx/logger.h>

#include <vector>
#include <string>

class AlgUtils {
public:
	template<typename NetType>
	static void SyncNet(const NetType& srcNet, NetType& dstNet, const float tau);

	template<typename NetType, typename OptimizerType>
	static void SaveModel(const NetType& net, const OptimizerType& opt, std::string path, log4cxx::LoggerPtr& logger);

	template<typename ModuleType>
	static void LoadModule(ModuleType& module, const std::string path, log4cxx::LoggerPtr& logger);

	template<typename NetType, typename OptimizerType>
	static void LoadModel(NetType& net, OptimizerType& opt, bool loadOptimizer, std::string path, log4cxx::LoggerPtr& logger);
};

//TODO: To check type of NetType
template<typename NetType>
void AlgUtils::SyncNet(const NetType& srcNet, NetType& dstNet, const float tau) {
	torch::NoGradGuard guard;

	auto paramDict = srcNet.named_parameters();
	auto buffDict = srcNet.named_buffers();
	auto targetParamDict = dstNet.named_parameters();
	auto targetBuffDict = dstNet.named_buffers();

	for (const auto& item: paramDict) {
		const auto& key = item.key();
		const auto param = item.value();
		auto& targetParam = targetParamDict[key];

		targetParam.mul_(1 - tau);
		targetParam.add_(param, tau);
	}

	for (const auto& item: buffDict) {
		const auto& key = item.key();
		const auto& buff = item.value();
		auto& targetBuff = targetBuffDict[key];

		targetBuff.mul_(1 - tau);
		targetBuff.add_(buff, tau);
	}
}

template<typename NetType, typename OptimizerType>
void AlgUtils::SaveModel(const NetType& net, const OptimizerType& opt, std::string path, log4cxx::LoggerPtr& logger) {
	torch::NoGradGuard guard;


	std::string modelPath = path + "_model.pt";
	torch::serialize::OutputArchive outputArchive;
	net.save(outputArchive);
	outputArchive.save_to(modelPath);
	LOG4CXX_INFO(logger, "Save model into " << modelPath);

	std::string optPath = path + "_optimizer.pt";
	torch::serialize::OutputArchive optimizerArchive;
	opt.save(optimizerArchive);
	optimizerArchive.save_to(optPath);
	LOG4CXX_INFO(logger, "Save optimizer into " << optPath);
}

template<typename ModuleType>
void AlgUtils::LoadModule(ModuleType& module, const std::string path, log4cxx::LoggerPtr& logger) {
	torch::serialize::InputArchive inChive;
	inChive.load_from(path);
	module.load(inChive);
	LOG4CXX_INFO(logger, "Load model from " << path);
}

template<typename NetType, typename OptimizerType>
void AlgUtils::LoadModel(NetType& net,  OptimizerType& opt, bool loadOptimizer, std::string path, log4cxx::LoggerPtr& logger) {
	torch::NoGradGuard guard;

	std::string modelPath = path + "_model.pt";
	LoadModule(net, modelPath, logger);

	if (loadOptimizer) {
		std::string optPath = path + "_optimizer.pt";
		LoadModule(opt, optPath, logger);
	}
}
#endif /* INC_ALG_UTILS_HPP_ */
