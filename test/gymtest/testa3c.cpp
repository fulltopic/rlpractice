/*
 * testa3c.cpp
 *
 *  Created on: Nov 15, 2021
 *      Author: zf
 */



#include "alg/a3c.hpp"

#include "a3c/a3ctcpserverhandle.hpp"
#include "a3c/a3ctcpserverconn.h"
#include "a3c/a3ctcpconfig.h"
#include "a3c/a3ctcpserverhdfacimpl.hpp"
#include "a3c/a3ctcpserver.h"

#include <iostream>
#include <string>
#include <memory>
#include <thread>


#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/simplelayout.h>
#include <log4cxx/logmanager.h>


#include "gymtest/env/airenv.h"
#include "gymtest/env/lunarenv.h"
#include "gymtest/airnets/aircnnnet.h"
#include "gymtest/airnets/airacbmnet.h"
#include "gymtest/airnets/airacnet.h"
#include "gymtest/lunarnets/cartacnet.h"
#include "gymtest/train/rawpolicy.h"
#include "gymtest/train/softmaxpolicy.h"
#include "gymtest/airnets/airachonet.h"

namespace {
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("testa2cn"));
const torch::Device deviceType = torch::kCUDA;

const int inputNum = 4;
//const int outputNum = 6;

void testServer0() {
	const int outputNum = 2;
	CartACFcNet net(inputNum, outputNum);
//	AirACCnnNet net(outputNum);

    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(0.001).eps(1e-3));
	boost::asio::io_service iio;

	std::shared_ptr<A3CTCPHandleFactory> factory
		= std::shared_ptr<A3CTCPHandleFactory>(
				new A3CTCPServerHdFacImpl<CartACFcNet, torch::optim::Adam>(iio, net, optimizer)
			);

	std::shared_ptr<A3CTCPServer> server = A3CTCPServer::Create(iio,factory);
	server->startAccept();

//	std::unique_ptr<std::thread> t = std::make_unique<std::thread>(
//		static_cast<std::size_t (boost::asio::io_context::*) ()>(&boost::asio::io_context::run), &iio);

	const int tNum = 2;
	std::vector<std::unique_ptr<std::thread>> ts;

	for (int i = 0; i < tNum; i ++) {
//	std::unique_ptr<std::thread> t = std::make_unique<std::thread>(
//		static_cast<std::size_t (boost::asio::io_context::*) ()>(&boost::asio::io_context::run), &iio);
	ts.push_back(std::make_unique<std::thread>(
			static_cast<std::size_t (boost::asio::io_context::*)()>(&boost::asio::io_context::run), &iio));
	}


    for (int i = 0; i < tNum; i ++) {
    	ts[i]->join();
    }

//	t->join();
}

void test0(const int batchSize, const int epochNum, const float entropyCoef, std::string serverAddr) {
	const int outputNum = 2;
	const std::string envName = "CartPole-v0";
//	const std::string envName = "CartPoleNoFrameskip-v4";

	const int num = batchSize;
//	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, num);
	env.init();
//	std::string testServerAddr = "tcp://127.0.0.1:10204";
//	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
//	LunarEnv testEnv(testServerAddr, envName, num);
//	testEnv.init();
//	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartACFcNet model(inputNum, outputNum);
	model.to(deviceType);
//	CartACFcNet targetModel(inputNum, outputNum);
//	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4};
    DqnOption option(inputShape, deviceType);

    option.isAtari = false;
    option.donePerEp = 1;
    option.multiLifes = false;
    option.statCap = batchSize * 2;
    option.entropyCoef = entropyCoef;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
    option.ppoLambda = 0.95;
    option.gamma = 0.99;
    option.statPathPrefix = "./a3c_cart0";
    option.saveModel = true;
    option.savePathPrefix = "./a3c_cart0";

    option.toTest = false;
    option.inputScale = 1;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.valueClip = false;
    option.normReward = false;
    option.loadModel = false;
    option.loadOptimizer = false;
    option.saveModel = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test17";
    option.tensorboardLogPath = "./logs/a3c_testcart0/tfevents.pb";
    option.gradSyncStep = 10;
    option.targetUpdateStep = 100;

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;

    /////////////////////////////////////// A3C
	boost::asio::io_service iio;

	auto client = A3CTCPClientHandle<CartACFcNet>::Create(iio, model);
	client->start();

	const int tNum = 1;
	std::vector<std::unique_ptr<std::thread>> ts;

	for (int i = 0; i < tNum; i ++) {
//	std::unique_ptr<std::thread> t = std::make_unique<std::thread>(
//		static_cast<std::size_t (boost::asio::io_context::*) ()>(&boost::asio::io_context::run), &iio);
	ts.push_back(std::make_unique<std::thread>(
			static_cast<std::size_t (boost::asio::io_context::*)()>(&boost::asio::io_context::run), &iio));
	}

	client->syncTarget();

    A3CNStep<CartACFcNet, LunarEnv, SoftmaxPolicy> a3c(model, env, env, policy, maxStep, option, client);
    a3c.train(epochNum, true);

    LOG4CXX_INFO(logger, "End of train");

    for (int i = 0; i < tNum; i ++) {
    	ts[i]->join();
    }
//    dqn.save();
}

void testServer1() {
	const int outputNum = 6;
//	CartACFcNet net(inputNum, outputNum);
	AirACHONet net(outputNum);

    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
	boost::asio::io_service iio;

	std::shared_ptr<A3CTCPHandleFactory> factory
		= std::shared_ptr<A3CTCPHandleFactory>(
				new A3CTCPServerHdFacImpl<AirACHONet, torch::optim::Adam>(iio, net, optimizer)
			);

	std::shared_ptr<A3CTCPServer> server = A3CTCPServer::Create(iio,factory);
	server->startAccept();

	const int tNum = 2;
	std::vector<std::unique_ptr<std::thread>> ts;

	for (int i = 0; i < tNum; i ++) {
//	std::unique_ptr<std::thread> t = std::make_unique<std::thread>(
//		static_cast<std::size_t (boost::asio::io_context::*) ()>(&boost::asio::io_context::run), &iio);
	ts.push_back(std::make_unique<std::thread>(
			static_cast<std::size_t (boost::asio::io_context::*)()>(&boost::asio::io_context::run), &iio));
	}


    for (int i = 0; i < tNum; i ++) {
    	ts[i]->join();
    }
}

void test1(const int batchSize, const int epochNum, const float entropyCoef, std::string serverAddr) {
	const int outputNum = 6;

	const std::string envName = "PongNoFrameskip-v4";
//	const std::string envName = "CartPoleNoFrameskip-v4";

	const int num = batchSize;
//	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, batchSize);
	env.init();
//	std::string testServerAddr = "tcp://127.0.0.1:10204";
//	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
//	LunarEnv testEnv(testServerAddr, envName, num);
//	testEnv.init();
//	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);
//	CartACFcNet targetModel(inputNum, outputNum);
//	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{batchSize, 4, 84, 84};
    DqnOption option(inputShape, deviceType);

    option.isAtari = true;
    option.donePerEp = 1;
    option.multiLifes = false;
    option.statCap = batchSize * 2;
    option.entropyCoef = entropyCoef;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
    option.ppoLambda = 0.95;
    option.gamma = 0.99;
    option.statPathPrefix = "./a3c_cart1";
    option.saveModel = true;
    option.savePathPrefix = "./a3c_cart1";

    option.toTest = false;
    option.inputScale = 1;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.valueClip = false;
    option.normReward = false;
    option.loadModel = false;
    option.loadOptimizer = false;
    option.saveModel = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test17";
    option.tensorboardLogPath = "./logs/a3c_testcart1/tfevents.pb";
    option.gradSyncStep = 10;
    option.targetUpdateStep = 100;

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 5;

    /////////////////////////////////////// A3C
	boost::asio::io_service iio;

	auto client = A3CTCPClientHandle<AirACHONet>::Create(iio, model);
	client->start();

	const int tNum = 1;
	std::vector<std::unique_ptr<std::thread>> ts;

	for (int i = 0; i < tNum; i ++) {
//	std::unique_ptr<std::thread> t = std::make_unique<std::thread>(
//		static_cast<std::size_t (boost::asio::io_context::*) ()>(&boost::asio::io_context::run), &iio);
	ts.push_back(std::make_unique<std::thread>(
			static_cast<std::size_t (boost::asio::io_context::*)()>(&boost::asio::io_context::run), &iio));
	}

	client->syncTarget();

    A3CNStep<AirACHONet, AirEnv, SoftmaxPolicy> a3c(model, env, env, policy, maxStep, option, client);
    a3c.train(epochNum, true);

    LOG4CXX_INFO(logger, "End of train");

    for (int i = 0; i < tNum; i ++) {
    	ts[i]->join();
    }
//    dqn.save();
}


//void testClient() {
//
//}
}

namespace {
void logConfigure(bool err) {
    log4cxx::ConsoleAppenderPtr appender(new log4cxx::ConsoleAppender());
    if (err) {
        appender->setTarget(LOG4CXX_STR("System.err"));
    }
    log4cxx::LayoutPtr layout(new log4cxx::SimpleLayout());
    appender->setLayout(layout);
    log4cxx::helpers::Pool pool;
    appender->activateOptions(pool);
    log4cxx::Logger::getRootLogger()->addAppender(appender);
//    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getDebug());
    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getInfo());
//    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getError());
    log4cxx::LogManager::getLoggerRepository()->setConfigured(true);
}
}

int main(int argc, char** argv) {
	logConfigure(false);

	if (atoi(argv[1]) == 1) {
		LOG4CXX_INFO(logger, "Start server");
		testServer1();
	} else {
		int batchSize = atoi(argv[2]);
		int epochNum = atoi(argv[3]);
		float entropyCoef = atof(argv[4]);
		std::string serverAddr(argv[5]);
		LOG4CXX_INFO(logger, "Start client: " << batchSize << ", " << epochNum << ", " << entropyCoef << " to " << serverAddr);

		test1(batchSize, epochNum, entropyCoef, serverAddr);
	}

	return 0;
}

