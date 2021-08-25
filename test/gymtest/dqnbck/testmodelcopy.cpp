/*
 * testmodelcopy.cpp
 *
 *  Created on: Apr 12, 2021
 *      Author: zf
 */


#include "alg/dqn.hpp"
#include "alg/nbrbdqn.hpp"

#include "gymtest/env/airenv.h"
#include "gymtest/airnets/aircnnnet.h"
#include "gymtest/train/rawpolicy.h"
#include "alg/dqnoption.h"

#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

namespace {
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("modelcpy"));

void getDict() {
	AirCnnNet model(18);
	model.to(torch::kCUDA);

	auto paramDict = model.named_parameters();
	auto buffDict = model.named_buffers();

	LOG4CXX_INFO(logger, "params");
	for (auto& item: paramDict) {
		LOG4CXX_INFO(logger, item.key() << ": " << item.value().sizes());
	}
	LOG4CXX_INFO(logger, "buffers");
	for (auto& item: buffDict) {
		LOG4CXX_INFO(logger, item.key() << ": " << item.value().sizes());
	}
}

void testCopy() {
	AirCnnNet model(18);
	model.to(torch::kCUDA);
	AirCnnNet targetNet(18);
	targetNet.to(torch::kCUDA);
	targetNet.eval();

	auto paramDict = model.named_parameters();
	auto buffDict = model.named_buffers();

	auto bias = paramDict["fc3.bias"];
	LOG4CXX_INFO(logger, "bias: " << bias);
	auto cBias = targetNet.named_parameters()["fc3.bias"];
	LOG4CXX_INFO(logger, "cBias new: " << cBias);
//	torch::Tensor newBias = torch::zeros(cBias.sizes()).to(torch::kCUDA);
//	cBias = newBias;
//	targetNet.named_parameters()["fc3.bias"] = cBias;
//	torch::_copy_from(cBias, bias);
	{
		torch::NoGradGuard gd;
		cBias.copy_(bias);
	}
	LOG4CXX_INFO(logger, "copied cbias: " << cBias);
	auto targetCBias = targetNet.named_parameters()["fc3.bias"];
	LOG4CXX_INFO(logger, "target cbias: " << targetCBias);
}
}

int main(int argc, char** argv) {
	log4cxx::BasicConfigurator::configure();
//	log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getDebug());
	log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getInfo());

//	getDict();
	testCopy();
//	test1(atoi(argv[1]), atoi(argv[2]));

	LOG4CXX_INFO(logger, "End of test");
}
