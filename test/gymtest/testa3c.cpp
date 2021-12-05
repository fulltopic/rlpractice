/*
 * testa3c.cpp
 *
 *  Created on: Nov 15, 2021
 *      Author: zf
 */



#include "alg/a3c.hpp"
#include "alg/a3ctest.hpp"
#include "alg/a2cnstep.hpp"
#include "alg/algtester.hpp"
#include "alg/a2cnstepgae.hpp"
#include "alg/a3cq.hpp"

#include "a3c/a3ctcpserverhandle.hpp"
#include "a3c/a3ctcpserverconn.h"
#include "a3c/a3ctcpconfig.h"
#include "a3c/a3ctcpserverhdfacimpl.hpp"
#include "a3c/a3ctcpserver.h"
#include "a3c/a3ctcpserverqhdfacimpl.hpp"
#include "a3c/a3cupdater.hpp"

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

//const int inputNum = 4;
//const int outputNum = 6;

void testServer0(const int batchSize, const int epNum, std::string logPath) {
	/////////////////////////////////////////////// Env
	const int inputNum = 4;
	const int outputNum = 2;
	CartACFcNet net(inputNum, outputNum);
	net.to(deviceType);
//	AirACCnnNet net(outputNum);
    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(0.001).eps(1e-3));

    SoftmaxPolicy policy(outputNum);



    ///////////////////////////////////////////////// Network
	boost::asio::io_service iio;

	std::shared_ptr<A3CTCPHandleFactory> factory
		= std::shared_ptr<A3CTCPHandleFactory>(
				new A3CTCPServerHdFacImpl<CartACFcNet, torch::optim::Adam>(iio, net, optimizer)
			);

	std::shared_ptr<A3CTCPServer> server = A3CTCPServer::Create(iio,factory);
//	server->startAccept();
	server->start();

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


    ////////////////////////////////////////////// Test
    at::IntArrayRef inputShape{batchSize, 4};
    DqnOption option(inputShape, deviceType);

    option.isAtari = false;
    option.donePerEp = 1;
    option.multiLifes = false;
    option.entropyCoef = 0; //0.01
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
    option.ppoLambda = 0.95;
    option.gamma = 0.99;
    option.saveModel = false;

    option.toTest = true;
    option.inputScale = 1;
    option.testGapEp = 100;
    option.testBatch = batchSize;
    option.testEp = epNum;
    option.tensorboardLogPath = logPath;

	const std::string envName = "CartPole-v0";
//	const std::string envName = "CartPoleNoFrameskip-v4";

	std::string serverAddr = "tcp://127.0.0.1:10210";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, batchSize);
	env.init();
	LOG4CXX_INFO(logger, "test env inited");

	AlgTester<CartACFcNet, LunarEnv, SoftmaxPolicy> tester(net, env, policy, option);
	uint64_t lastUpdateNum = server->getUpdateNum();

    const int pollMinute = 3;
    server->setPollMinute(pollMinute);

    while (true) {
    	sleep(pollMinute);

    	auto updateNum = server->getUpdateNum();
    	if (updateNum - lastUpdateNum > option.testGapEp) {
    		tester.test();
    		lastUpdateNum = updateNum;
    	}
    }

    ///////////////////////////////////////////// Join
    for (int i = 0; i < tNum; i ++) {
    	ts[i]->join();
    }

//	t->join();
}

void test0(const int batchSize, const int epochNum, const float entropyCoef,
		std::string serverAddr, std::string logPath) {
	const int inputNum = 4;
	const int outputNum = 2;
    const int maxStep = 8;
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
    option.entropyCoef = entropyCoef; //0.01
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
    option.ppoLambda = 0.95;
    option.gamma = 0.99;
//    option.statPathPrefix = "./a3c_cart0";
    option.saveModel = true;
//    option.savePathPrefix = "./a3c_cart0";

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
//    option.tensorboardLogPath = "./logs/a3c_testcart0/tfevents.pb";
    option.tensorboardLogPath = logPath;
    option.gradSyncStep = maxStep * 4;
    option.targetUpdateStep = maxStep * 4;

    SoftmaxPolicy policy(outputNum);

    /////////////////////////////////////// A3C
	boost::asio::io_service iio;

	auto client = A3CTCPClientHandle<CartACFcNet>::Create(iio, model);
	client->start();

	const int tNum = 1;
	std::vector<std::unique_ptr<std::thread>> ts;

	for (int i = 0; i < tNum; i ++) {
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


void testServer1(const int batchSize, const int epNum, std::string logPath) {
	/////////////////////////////////////////////// Env
	const int outputNum = 6;

	const std::string envName = "PongNoFrameskip-v4";

	std::string serverAddr = "tcp://127.0.0.1:10210";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, batchSize);
	env.init();
	LOG4CXX_INFO(logger, "test env inited");

	AirACHONet net(outputNum);
	net.to(deviceType);
    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(0.001).eps(5e-4));

    SoftmaxPolicy policy(outputNum);



    ///////////////////////////////////////////////// Network
	boost::asio::io_service iio;

	std::shared_ptr<A3CTCPHandleFactory> factory
		= std::shared_ptr<A3CTCPHandleFactory>(
				new A3CTCPServerHdFacImpl<AirACHONet, torch::optim::Adam>(iio, net, optimizer)
			);

	std::shared_ptr<A3CTCPServer> server = A3CTCPServer::Create(iio,factory);
	server->start();


	const int tNum = 2;
	std::vector<std::unique_ptr<std::thread>> ts;

	for (int i = 0; i < tNum; i ++) {
		ts.push_back(std::make_unique<std::thread>(
			static_cast<std::size_t (boost::asio::io_context::*)()>(&boost::asio::io_context::run), &iio));
	}


    ////////////////////////////////////////////// Test
    at::IntArrayRef inputShape{batchSize, 4, 84, 84};
    DqnOption option(inputShape, deviceType);

    option.isAtari = true;
    option.donePerEp = 1;
    option.multiLifes = false;
//    option.entropyCoef = 0; //0.01
//    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
//    option.gamma = 0.99;

    option.toTest = true;
    option.inputScale = 255;
    option.testGapEp = 100;
    option.testBatch = batchSize;
    option.testEp = epNum;
    option.tensorboardLogPath = logPath;

    option.saveModel = false;


	AlgTester<AirACHONet, AirEnv, SoftmaxPolicy> tester(net, env, policy, option);
	uint64_t lastUpdateNum = server->getUpdateNum();

    const int pollMinute = 5;
    server->setPollMinute(pollMinute);

    while (true) {
    	sleep(pollMinute);

    	auto updateNum = server->getUpdateNum();
    	if (updateNum - lastUpdateNum > option.testGapEp) {
    		tester.test();
    		lastUpdateNum = updateNum;
    	}
    }

    ///////////////////////////////////////////// Join
    for (int i = 0; i < tNum; i ++) {
    	ts[i]->join();
    }

}

//TODO: decrease sync gap
void test1(const int batchSize, const int epochNum, const float entropyCoef, std::string serverAddr, std::string logPath) {
	const int outputNum = 6;
	const std::string envName = "PongNoFrameskip-v4";
	const int num = batchSize;
    const int maxStep = 8;

	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, batchSize);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);
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
//    option.statPathPrefix = "./a3c_pong0";
//    option.saveModel = false;
    option.savePathPrefix = "./a3c_pong1";

    option.toTest = false;
    option.inputScale = 255;
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
//    option.tensorboardLogPath = "./logs/a3c_testpong0/tfevents.pb";
    option.tensorboardLogPath = logPath;
    option.gradSyncStep = maxStep * 2;
    option.targetUpdateStep = maxStep * 2;

    SoftmaxPolicy policy(outputNum);

    /////////////////////////////////////// A3C
	boost::asio::io_service iio;

	auto client = A3CTCPClientHandle<AirACHONet>::Create(iio, model);
	client->start();

	const int tNum = 1;
	std::vector<std::unique_ptr<std::thread>> ts;

	for (int i = 0; i < tNum; i ++) {
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


void testServer2(const int batchSize, const int epNum, std::string logPath) {
	/////////////////////////////////////////////// Env
	const int outputNum = 6;

	const std::string envName = "PongNoFrameskip-v4";

	std::string serverAddr = "tcp://127.0.0.1:10210";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, batchSize);
	env.init();
	LOG4CXX_INFO(logger, "test env inited");

	AirACHONet net(outputNum);
	net.to(deviceType);
    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(0.0001).eps(5e-4));

    SoftmaxPolicy policy(outputNum);



    ///////////////////////////////////////////////// Network
	boost::asio::io_service iio;

	std::shared_ptr<A3CTCPHandleFactory> factory
		= std::shared_ptr<A3CTCPHandleFactory>(
				new A3CTCPServerHdFacImpl<AirACHONet, torch::optim::Adam>(iio, net, optimizer)
			);

	std::shared_ptr<A3CTCPServer> server = A3CTCPServer::Create(iio,factory);
	server->start();


	const int tNum = 2;
	std::vector<std::unique_ptr<std::thread>> ts;

	for (int i = 0; i < tNum; i ++) {
		ts.push_back(std::make_unique<std::thread>(
			static_cast<std::size_t (boost::asio::io_context::*)()>(&boost::asio::io_context::run), &iio));
	}


    ////////////////////////////////////////////// Test
    at::IntArrayRef inputShape{batchSize, 4, 84, 84};
    DqnOption option(inputShape, deviceType);

    option.isAtari = true;
    option.donePerEp = 1;
    option.multiLifes = false;
//    option.entropyCoef = 0; //0.01
//    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
//    option.gamma = 0.99;

    option.toTest = true;
    option.inputScale = 255;
    option.testGapEp = 100;
    option.testBatch = batchSize;
    option.testEp = epNum;
    option.tensorboardLogPath = logPath;

    option.saveModel = false;


	AlgTester<AirACHONet, AirEnv, SoftmaxPolicy> tester(net, env, policy, option);
	uint64_t lastUpdateNum = server->getUpdateNum();

    const int pollMinute = 5;
    server->setPollMinute(pollMinute);

    while (true) {
    	sleep(pollMinute);

    	auto updateNum = server->getUpdateNum();
    	if (updateNum - lastUpdateNum > option.testGapEp) {
    		tester.test();
    		lastUpdateNum = updateNum;
    	}
    }

    ///////////////////////////////////////////// Join
    for (int i = 0; i < tNum; i ++) {
    	ts[i]->join();
    }

}

//TODO: decrease sync gap
void test2(const int batchSize, const int epochNum, const float entropyCoef, std::string serverAddr, std::string logPath) {
	const int outputNum = 6;
	const std::string envName = "PongNoFrameskip-v4";
	const int num = batchSize;
    const int maxStep = 5;

	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, batchSize);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);
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
//    option.statPathPrefix = "./a3c_pong0";
//    option.saveModel = false;
    option.savePathPrefix = "./a3c_pong1";

    option.toTest = false;
    option.inputScale = 255;
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
//    option.tensorboardLogPath = "./logs/a3c_testpong0/tfevents.pb";
    option.tensorboardLogPath = logPath;
    option.gradSyncStep = maxStep * 2;
    option.targetUpdateStep = maxStep * 2;

    SoftmaxPolicy policy(outputNum);

    /////////////////////////////////////// A3C
	boost::asio::io_service iio;

	auto client = A3CTCPClientHandle<AirACHONet>::Create(iio, model);
	client->start();

	const int tNum = 1;
	std::vector<std::unique_ptr<std::thread>> ts;

	for (int i = 0; i < tNum; i ++) {
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


void testServer3(const int batchSize, const int epNum, std::string logPath) {
	/////////////////////////////////////////////// Env
	const int outputNum = 6;

	const std::string envName = "PongNoFrameskip-v4";

	std::string serverAddr = "tcp://127.0.0.1:10210";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, batchSize);
	env.init();
	LOG4CXX_INFO(logger, "test env inited");

	AirACHONet net(outputNum);
	net.to(deviceType);
    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(0.001).eps(5e-4));

    SoftmaxPolicy policy(outputNum);



    ///////////////////////////////////////////////// Network
	boost::asio::io_service iio;

	std::shared_ptr<A3CTCPHandleFactory> factory
		= std::shared_ptr<A3CTCPHandleFactory>(
				new A3CTCPServerHdFacImpl<AirACHONet, torch::optim::Adam>(iio, net, optimizer)
			);

	std::shared_ptr<A3CTCPServer> server = A3CTCPServer::Create(iio,factory);
	server->start();


	const int tNum = 2;
	std::vector<std::unique_ptr<std::thread>> ts;

	for (int i = 0; i < tNum; i ++) {
		ts.push_back(std::make_unique<std::thread>(
			static_cast<std::size_t (boost::asio::io_context::*)()>(&boost::asio::io_context::run), &iio));
	}


    ////////////////////////////////////////////// Test
    at::IntArrayRef inputShape{batchSize, 4, 84, 84};
    DqnOption option(inputShape, deviceType);

    option.isAtari = true;
    option.donePerEp = 1;
    option.multiLifes = false;
//    option.entropyCoef = 0; //0.01
//    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
//    option.gamma = 0.99;

    option.toTest = true;
    option.inputScale = 255;
    option.testGapEp = 100;
    option.testBatch = batchSize;
    option.testEp = epNum;
    option.tensorboardLogPath = logPath;

    option.saveModel = false;


	AlgTester<AirACHONet, AirEnv, SoftmaxPolicy> tester(net, env, policy, option);
	uint64_t lastUpdateNum = server->getUpdateNum();

    const int pollMinute = 5;
    server->setPollMinute(pollMinute);

    while (true) {
    	sleep(pollMinute);

    	auto updateNum = server->getUpdateNum();
    	if (updateNum - lastUpdateNum > option.testGapEp) {
    		tester.test();
    		lastUpdateNum = updateNum;
    	}
    }

    ///////////////////////////////////////////// Join
    for (int i = 0; i < tNum; i ++) {
    	ts[i]->join();
    }

}

//TODO: decrease sync gap
void test3(const int batchSize, const int epochNum, const float entropyCoef, std::string serverAddr, std::string logPath) {
	const int outputNum = 6;
	const std::string envName = "PongNoFrameskip-v4";
	const int num = batchSize;
    const int maxStep = 5;

	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, batchSize);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);
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
//    option.statPathPrefix = "./a3c_pong0";
//    option.saveModel = false;
    option.savePathPrefix = "./a3c_pong1";

    option.toTest = false;
    option.inputScale = 255;
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
//    option.tensorboardLogPath = "./logs/a3c_testpong0/tfevents.pb";
    option.tensorboardLogPath = logPath;
    option.gradSyncStep = maxStep * 2;
    option.targetUpdateStep = maxStep * 2;

    SoftmaxPolicy policy(outputNum);

    /////////////////////////////////////// A3C
	boost::asio::io_service iio;

	auto client = A3CTCPClientHandle<AirACHONet>::Create(iio, model);
	client->start();

	const int tNum = 1;
	std::vector<std::unique_ptr<std::thread>> ts;

	for (int i = 0; i < tNum; i ++) {
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


void testServer4(const int batchSize, const int epNum, std::string logPath, std::string qLogPath) {
	/////////////////////////////////////////////// Env
	const int outputNum = 6;

	const std::string envName = "PongNoFrameskip-v4";

	std::string serverAddr = "tcp://127.0.0.1:10210";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, batchSize);
	env.init();
	LOG4CXX_INFO(logger, "test env inited");

	AirACHONet net(outputNum);
	net.to(deviceType);
    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(0.001).eps(5e-4));

    SoftmaxPolicy policy(outputNum);



    ///////////////////////////////////////////////// Network
    const int maxThreshold = 20;
    const int qLogGap = 16;

    A3CGradQueue q(maxThreshold, qLogGap, qLogPath);

	boost::asio::io_service iio;

	std::shared_ptr<A3CTCPHandleFactory> factory
		= std::shared_ptr<A3CTCPHandleFactory>(
				new A3CTCPServerQHdFacImpl<AirACHONet>(iio, net, q)
			);

	std::shared_ptr<A3CTCPServer> server = A3CTCPServer::Create(iio,factory);
	server->start();


	const int tNum = 2;
	std::vector<std::unique_ptr<std::thread>> ts;

	for (int i = 0; i < tNum; i ++) {
		ts.push_back(std::make_unique<std::thread>(
			static_cast<std::size_t (boost::asio::io_context::*)()>(&boost::asio::io_context::run), &iio));
	}

	A3CNetUpdater<AirACHONet, torch::optim::Adam> updater(net, optimizer, q);
	std::thread updateT(&A3CNetUpdater<AirACHONet, torch::optim::Adam>::processGrad, &updater);


    ////////////////////////////////////////////// Test
    at::IntArrayRef inputShape{batchSize, 4, 84, 84};
    DqnOption option(inputShape, deviceType);

    option.isAtari = true;
    option.donePerEp = 1;
    option.multiLifes = false;
//    option.entropyCoef = 0; //0.01
//    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
//    option.gamma = 0.99;

    option.toTest = true;
    option.inputScale = 255;
    option.testGapEp = 100;
    option.testBatch = batchSize;
    option.testEp = epNum;
    option.tensorboardLogPath = logPath;

    option.saveModel = false;


	AlgTester<AirACHONet, AirEnv, SoftmaxPolicy> tester(net, env, policy, option);
	uint64_t lastUpdateNum = server->getUpdateNum();

    const int pollMinute = 5;
    server->setPollMinute(pollMinute);

    while (true) {
    	sleep(pollMinute);

    	auto updateNum = server->getUpdateNum();
    	if (updateNum - lastUpdateNum > option.testGapEp) {
    		tester.test();
    		lastUpdateNum = updateNum;
    	}
    }

    ///////////////////////////////////////////// Join
    for (int i = 0; i < tNum; i ++) {
    	ts[i]->join();
    }
    updateT.join();

}

//TODO: decrease sync gap
void test4(const int batchSize, const int epochNum, const float entropyCoef, std::string serverAddr, std::string logPath) {
	const int outputNum = 6;
	const std::string envName = "PongNoFrameskip-v4";
	const int num = batchSize;
    const int maxStep = 5;

	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, batchSize);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);
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
//    option.statPathPrefix = "./a3c_pong0";
//    option.saveModel = false;
    option.savePathPrefix = "./a3c_pong1";

    option.toTest = false;
    option.inputScale = 255;
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
//    option.tensorboardLogPath = "./logs/a3c_testpong0/tfevents.pb";
    option.tensorboardLogPath = logPath;
    option.gradSyncStep = maxStep * 2; //useless
    option.targetUpdateStep = maxStep * 4;

    SoftmaxPolicy policy(outputNum);

    /////////////////////////////////////// A3C
	boost::asio::io_service iio;

	auto client = A3CTCPClientHandle<AirACHONet>::Create(iio, model);
	client->start();

	const int tNum = 1;
	std::vector<std::unique_ptr<std::thread>> ts;

	for (int i = 0; i < tNum; i ++) {
	ts.push_back(std::make_unique<std::thread>(
			static_cast<std::size_t (boost::asio::io_context::*)()>(&boost::asio::io_context::run), &iio));
	}

	client->syncTarget();

	A3CQNStep<AirACHONet, AirEnv, SoftmaxPolicy> a3c(model, env, env, policy, maxStep, option, client);
    a3c.train(epochNum, true);

    LOG4CXX_INFO(logger, "End of train");

    for (int i = 0; i < tNum; i ++) {
    	ts[i]->join();
    }
//    dqn.save();
}

void testServer5(const int batchSize, const int epNum, std::string logPath) {
	/////////////////////////////////////////////// Env
	const int outputNum = 6;

	const std::string envName = "PongNoFrameskip-v4";

	std::string serverAddr = "tcp://127.0.0.1:10210";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, batchSize);
	env.init();
	LOG4CXX_INFO(logger, "test env inited");

	AirACHONet net(outputNum);
	net.to(deviceType);
    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(0.0002).eps(5e-4));

    SoftmaxPolicy policy(outputNum);



    ///////////////////////////////////////////////// Network
	boost::asio::io_service iio;

	std::shared_ptr<A3CTCPHandleFactory> factory
		= std::shared_ptr<A3CTCPHandleFactory>(
				new A3CTCPServerHdFacImpl<AirACHONet, torch::optim::Adam>(iio, net, optimizer)
			);

	std::shared_ptr<A3CTCPServer> server = A3CTCPServer::Create(iio,factory);
	server->start();

	//TODO: server coredump, to make net shared
	const int tNum = 2;
	std::vector<std::unique_ptr<std::thread>> ts;

	for (int i = 0; i < tNum; i ++) {
		ts.push_back(std::make_unique<std::thread>(
			static_cast<std::size_t (boost::asio::io_context::*)()>(&boost::asio::io_context::run), &iio));
	}


    ////////////////////////////////////////////// Test
    at::IntArrayRef inputShape{batchSize, 4, 84, 84};
    DqnOption option(inputShape, deviceType);

    option.isAtari = true;
    option.donePerEp = 1;
    option.multiLifes = false;
//    option.entropyCoef = 0; //0.01
//    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
//    option.gamma = 0.99;

    option.toTest = true;
    option.inputScale = 255;
    option.testGapEp = 100;
    option.testBatch = batchSize;
    option.testEp = epNum;
    option.tensorboardLogPath = logPath;

    option.saveModel = false;


	AlgTester<AirACHONet, AirEnv, SoftmaxPolicy> tester(net, env, policy, option);
	uint64_t lastUpdateNum = server->getUpdateNum();

    const int pollMinute = 5;
    server->setPollMinute(pollMinute);

    while (true) {
    	sleep(pollMinute);

    	auto updateNum = server->getUpdateNum();
    	if (updateNum - lastUpdateNum > option.testGapEp) {
    		tester.test();
    		lastUpdateNum = updateNum;
    	}
    }

    ///////////////////////////////////////////// Join
    for (int i = 0; i < tNum; i ++) {
    	ts[i]->join();
    }

}

//0.01, 0.01, 0.05, 0.005
//TODO: decrease sync gap
void test5(const int batchSize, const int epochNum, const float entropyCoef, std::string serverAddr, std::string logPath) {
	const int outputNum = 6;
	const std::string envName = "PongNoFrameskip-v4";
	const int num = batchSize;
    const int maxStep = 5;

	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, batchSize);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);
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
//    option.statPathPrefix = "./a3c_pong0";
//    option.saveModel = false;
    option.savePathPrefix = "./a3c_pong1";

    option.toTest = false;
    option.inputScale = 255;
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
//    option.tensorboardLogPath = "./logs/a3c_testpong0/tfevents.pb";
    option.tensorboardLogPath = logPath;
    option.gradSyncStep = maxStep * 2;
    option.targetUpdateStep = maxStep * 2;

    SoftmaxPolicy policy(outputNum);

    /////////////////////////////////////// A3C
	boost::asio::io_service iio;

	auto client = A3CTCPClientHandle<AirACHONet>::Create(iio, model);
	client->start();

	const int tNum = 1;
	std::vector<std::unique_ptr<std::thread>> ts;

	for (int i = 0; i < tNum; i ++) {
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


void testServer6(const int batchSize, const int epNum, std::string logPath, std::string qLogPath) {
	/////////////////////////////////////////////// Env
	const int outputNum = 6;

	const std::string envName = "PongNoFrameskip-v4";

	std::string serverAddr = "tcp://127.0.0.1:10210";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, batchSize);
	env.init();
	LOG4CXX_INFO(logger, "test env inited");

	AirACHONet net(outputNum);
	net.to(deviceType);
    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(0.0005).eps(5e-4));

    SoftmaxPolicy policy(outputNum);



    ///////////////////////////////////////////////// Network
    const int maxThreshold = 20;
    const int qLogGap = 16;

    A3CGradQueue q(maxThreshold, qLogGap, qLogPath);

	boost::asio::io_service iio;

	std::shared_ptr<A3CTCPHandleFactory> factory
		= std::shared_ptr<A3CTCPHandleFactory>(
				new A3CTCPServerQHdFacImpl<AirACHONet>(iio, net, q)
			);

	std::shared_ptr<A3CTCPServer> server = A3CTCPServer::Create(iio,factory);
	server->start();


	const int tNum = 2;
	std::vector<std::unique_ptr<std::thread>> ts;

	for (int i = 0; i < tNum; i ++) {
		ts.push_back(std::make_unique<std::thread>(
			static_cast<std::size_t (boost::asio::io_context::*)()>(&boost::asio::io_context::run), &iio));
	}

	A3CNetUpdater<AirACHONet, torch::optim::Adam> updater(net, optimizer, q);
	std::thread updateT(&A3CNetUpdater<AirACHONet, torch::optim::Adam>::processGrad, &updater);


    ////////////////////////////////////////////// Test
    at::IntArrayRef inputShape{batchSize, 4, 84, 84};
    DqnOption option(inputShape, deviceType);

    option.isAtari = true;
    option.donePerEp = 1;
    option.multiLifes = false;
//    option.entropyCoef = 0; //0.01
//    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
//    option.gamma = 0.99;

    option.toTest = true;
    option.inputScale = 255;
    option.testGapEp = 100;
    option.testBatch = batchSize;
    option.testEp = epNum;
    option.tensorboardLogPath = logPath;

    option.saveModel = false;


	AlgTester<AirACHONet, AirEnv, SoftmaxPolicy> tester(net, env, policy, option);
	uint64_t lastUpdateNum = server->getUpdateNum();

    const int pollMinute = 5;
    server->setPollMinute(pollMinute);

    while (true) {
    	sleep(pollMinute);

    	auto updateNum = server->getUpdateNum();
    	if (updateNum - lastUpdateNum > option.testGapEp) {
    		tester.test();
    		lastUpdateNum = updateNum;
    	}
    }

    ///////////////////////////////////////////// Join
    for (int i = 0; i < tNum; i ++) {
    	ts[i]->join();
    }
    updateT.join();

}

//TODO: decrease sync gap
void test6(const int batchSize, const int epochNum, const float entropyCoef, std::string serverAddr, std::string logPath) {
	const int outputNum = 6;
	const std::string envName = "PongNoFrameskip-v4";
	const int num = batchSize;
    const int maxStep = 5;

	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, batchSize);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);
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
//    option.statPathPrefix = "./a3c_pong0";
//    option.saveModel = false;
    option.savePathPrefix = "./a3c_pong1";

    option.toTest = false;
    option.inputScale = 255;
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
//    option.tensorboardLogPath = "./logs/a3c_testpong0/tfevents.pb";
    option.tensorboardLogPath = logPath;
    option.gradSyncStep = maxStep; //useless
    option.targetUpdateStep = maxStep * 4;

    SoftmaxPolicy policy(outputNum);

    /////////////////////////////////////// A3C
	boost::asio::io_service iio;

	auto client = A3CTCPClientHandle<AirACHONet>::Create(iio, model);
	client->start();

	const int tNum = 1;
	std::vector<std::unique_ptr<std::thread>> ts;

	for (int i = 0; i < tNum; i ++) {
	ts.push_back(std::make_unique<std::thread>(
			static_cast<std::size_t (boost::asio::io_context::*)()>(&boost::asio::io_context::run), &iio));
	}

	client->syncTarget();

	A3CQNStep<AirACHONet, AirEnv, SoftmaxPolicy> a3c(model, env, env, policy, maxStep, option, client);
    a3c.train(epochNum, true);

    LOG4CXX_INFO(logger, "End of train");

    for (int i = 0; i < tNum; i ++) {
    	ts[i]->join();
    }
//    dqn.save();
}



void testasync(const int updateNum) {
	const int inputNum = 4;
	const int outputNum = 2;
    const int maxStep = 5;
	const std::string envName = "CartPole-v0";
	const int batchSize = 32;
//	const float entropyCoef = 0.05;
//	const std::string envName = "CartPoleNoFrameskip-v4";

	const int num = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10203";
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
	CartACFcNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(targetModel.parameters(), torch::optim::AdamOptions(0.001));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4};
    DqnOption option(inputShape, deviceType);

    option.isAtari = false;
    option.donePerEp = 1;
    option.multiLifes = false;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 1;
    option.maxGradNormClip = 0.1;
//    option.ppoLambda = 0.95;
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
    option.tensorboardLogPath = "./logs/a3c_testcartasync/tfevents.pb";
    option.gradSyncStep = maxStep;
    option.targetUpdateStep = maxStep;

    SoftmaxPolicy policy(outputNum);

    A3CNStepTest<CartACFcNet, torch::optim::Adam, LunarEnv, SoftmaxPolicy> a3c(
    		model, targetModel, optimizer, env, env, policy, maxStep, option);
    a3c.train(updateNum, true);
}

void testasyncpong(const int updateNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int batchSize = 32;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);
	AirACHONet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(targetModel.parameters(),
    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = batchSize;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.tensorboardLogPath = "./logs/a3c_testasyncpong/tfevents.pb";
    option.statPathPrefix = "./a3c_testasyncpong";
    option.saveModel = false;
    option.savePathPrefix = "./ponga2cnbatch_test30";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;
    option.logInterval = 100;

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A3CNStepTest<AirACHONet, torch::optim::RMSprop, AirEnv, SoftmaxPolicy> a3c(
    		model, targetModel, optimizer, env, env, policy, maxStep, option);
    a3c.train(updateNum, true);
}


void testa2c(const int updateNum) {
	const int inputNum = 4;
	const int outputNum = 2;
    const int maxStep = 5;
	const std::string envName = "CartPole-v0";
	const int batchSize = 32;
	const int testEnvNum = 8;
//	const float entropyCoef = 0.05;
//	const std::string envName = "CartPoleNoFrameskip-v4";

	const int num = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10208";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	LunarEnv testEnv(testServerAddr, envName, testEnvNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartACFcNet model(inputNum, outputNum);
	model.to(deviceType);
//	CartACFcNet targetModel(inputNum, outputNum);
//	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4};
    DqnOption option(inputShape, deviceType);

    option.isAtari = false;
    option.donePerEp = 1;
    option.multiLifes = false;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.05;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
//    option.ppoLambda = 0.95;
    option.gamma = 0.99;
//    option.statPathPrefix = "./a3c_cart0";
    option.saveModel = false;
    option.savePathPrefix = "./a3c_cart0";

    option.toTest = true;
    option.testGapEp = 1000;
    option.testBatch = testEnvNum;
    option.testEp = testEnvNum;
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
    option.tensorboardLogPath = "./logs/a3c_testcarta2c/tfevents.pb";
    option.gradSyncStep = maxStep;
    option.targetUpdateStep = maxStep;

    SoftmaxPolicy policy(outputNum);

//    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);

    A2CNStep<CartACFcNet, LunarEnv, SoftmaxPolicy, torch::optim::Adam> a2c(
    		model, env, testEnv, policy, optimizer, maxStep, option);
    a2c.train(updateNum, true);
}

void testa2cpong(int updateNum) {
    const int maxStep = 5;

	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int batchSize = 50;
	const int clientNum = batchSize;
	const int testClientNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10206";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3).eps(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = batchSize;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;

    option.toTest = true;
    option.testBatch = testClientNum;
    option.testEp = testClientNum;
    option.testGapEp = 2000;
    option.multiLifes = false;

    option.tensorboardLogPath = "./logs/a3c_testa2cpong/tfevents.pb";
    option.logInterval = maxStep * 10;

    option.loadModel = false;
    option.loadOptimizer = false;

    option.statPathPrefix = "./a3c_testa2cpong";
    option.saveModel = true;
    option.saveThreshold = -20;
    option.saveStep = 1;
    option.savePathPrefix = "./a3c_testa2cpong";

    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, testEnv, policy, optimizer, maxStep, option);
    a2c.train(updateNum, true);
}

void testa2cgaepong(int updateNum) {
    const int maxStep = 10;

	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int batchSize = 50;
	const int clientNum = batchSize;
	const int testClientNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3).eps(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = batchSize;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
    option.statPathPrefix = "./a3c_a2cgaepong";
    option.saveModel = false;
    option.savePathPrefix = "./a3c_a2cgaepong";
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.statCap = 128;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;

    option.toTest = true;
    option.testBatch = testClientNum;
    option.testEp = testClientNum;
    option.testGapEp = 2000;
    option.multiLifes = false;

    option.tensorboardLogPath = "./logs/a3c_testa2cpong/tfevents.pb";
    option.logInterval = maxStep * 10;

    option.loadModel = false;
    option.loadOptimizer = false;

    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    A2CNStepGae<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, testEnv, policy, optimizer, maxStep, option);
    a2c.train(updateNum);
}
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
		int batchSize = atoi(argv[2]);
		int epNum = atoi(argv[3]);
		std::string logPath(argv[4]);
		std::string qLogPath(argv[5]);
		LOG4CXX_INFO(logger, "Start server");
//		testServer5(batchSize, epNum, logPath);
		testServer6(batchSize, epNum, logPath, qLogPath);
	} else {
		int batchSize = atoi(argv[2]);
		int epochNum = atoi(argv[3]);
		float entropyCoef = atof(argv[4]);
		std::string serverAddr(argv[5]);
		std::string logPath(argv[6]);
		LOG4CXX_INFO(logger, "Start client: " << batchSize << ", " << epochNum << ", " << entropyCoef << " to " << serverAddr);

		test6(batchSize, epochNum, entropyCoef, serverAddr, logPath);
	}

//	testasync(atoi(argv[1]));
//	testa2c(atoi(argv[1]));
//	testa2cpong(atoi(argv[1]));
//	testa2cgaepong(atoi(argv[1]));
//	testasyncpong(atoi(argv[1]));

	return 0;
}

