/*
 * testappo.cpp
 *
 *  Created on: Dec 15, 2021
 *      Author: zf
 */



#include "alg/appo/appodataq.h"
#include "alg/appo/appoupdater.hpp"
#include "alg/appo/appoworker.hpp"
//#include "alg/a2cnstep.hpp"
#include "alg/utils/algtester.hpp"
#include "alg/a2cnstepgae.hpp"
#include "alg/a3c/a3cq.hpp"
#include "alg/a3c/a3cgradshared.hpp"

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

void test0(const int clientNum, const int roundNum, const int maxStep, const int epochNum, const int updateNum) {
	//Env Config
	const int inputNum = 4;
	const int outputNum = 2;
//	const int maxStep = 8;
	const std::string envName = "CartPole-v0";


	//Training Config
//	const int roundNum = 1;
//    const int maxStep = 5;
    const int workerNum = roundNum;
    std::vector<float> entropyCoefs {0.01, 0.01, 0.005, 0.02, 0,02};
    assert(workerNum <= entropyCoefs.size());

    const int basePort = 10205;
    const int testPort = 10210;
    const std::string addrBase = "tcp://127.0.0.1:";
    const std::string logBase = "./logs/appo_testcart1/";
    const std::string logFileName = "tfevents.pb";

    //Net
    CartACFcNet model(inputNum, outputNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.0003));
    LOG4CXX_INFO(logger, "Model ready");
    SoftmaxPolicy policy(outputNum);
    std::mutex updateMutex;

	//Queue
	AsyncPPODataQ q(32);

	//Workers
	at::IntArrayRef inputShape{clientNum, 4};

    std::vector<LunarEnv*> envs;
    std::vector<APPOWorker<CartACFcNet, LunarEnv, SoftmaxPolicy>*> workers;
    for(int i = 0; i < workerNum; i ++) {
    	DqnOption option(inputShape, deviceType);

    	option.isAtari = false;
    	option.donePerEp = 1;
    	option.multiLifes = false;
    	option.gamma = 0.99;
    	option.ppoLambda = 0.95;
    	option.inputScale = 1;
//    	option.batchSize = batchSize;
    	option.envNum = clientNum;
    	option.trajStepNum = maxStep;
    	option.rewardScale = 1;
    	option.rewardMin = -1;
    	option.rewardMax = 1;
    	option.tensorboardLogPath = logBase + std::to_string(i) + "/" + logFileName;
    	LOG4CXX_INFO(logger, "To put log in" << option.tensorboardLogPath);

//    	options.push_back(option);

    	std::string serverAddr = addrBase + std::to_string(basePort + i);
    	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
    	LunarEnv* env = new LunarEnv(serverAddr, envName, clientNum);
    	env->init();
    	envs.push_back(env);
    	LOG4CXX_INFO(logger, "Env " << envName << " " << i << " ready");

    	APPOWorker<CartACFcNet, LunarEnv, SoftmaxPolicy>* worker =
    			new APPOWorker<CartACFcNet, LunarEnv, SoftmaxPolicy> (
    					model,
						*env,
						policy,
						option,
						q,
						outputNum
    					);
    	workers.push_back(worker);
    }

	std::vector<std::unique_ptr<std::thread>> ts;

	for (int i = 0; i < workerNum; i ++) {
		ts.push_back(std::make_unique<std::thread>(
			&APPOWorker<CartACFcNet, LunarEnv, SoftmaxPolicy>::train, &(*workers[i]), updateNum));
	}

	////////////////////////////////////// Updater ////////////////////////////////////////
	at::IntArrayRef updateInputShape{clientNum, 4};

	DqnOption updateOption(updateInputShape, deviceType);
	updateOption.isAtari = false;
	updateOption.valueCoef = 0.25;
	updateOption.maxGradNormClip = 0.1;
	updateOption.gamma = 0.99;
	updateOption.ppoLambda = 0.95;
	updateOption.ppoEpsilon = 0.1;
	updateOption.appoRoundNum = roundNum;
	updateOption.epochNum = epochNum;
	updateOption.trajStepNum = maxStep * clientNum * roundNum;
	updateOption.inputScale = 1;
	updateOption.rewardScale = 1;
	updateOption.rewardMin = -1;
	updateOption.rewardMax = 1;
	updateOption.tensorboardLogPath = logBase + "update" + "/" + logFileName;

	APPOUpdater<CartACFcNet, torch::optim::Adam> updater (
			model,
			optimizer,
			updateOption,
			q,
			outputNum
			);

	auto updateThread = std::make_unique<std::thread>(
			&APPOUpdater<CartACFcNet, torch::optim::Adam>::train, &updater, updateNum);


	////////////////////////////////////// Test/////////////////////////////////////////
    const int pollMinute = 1 * 60;
    const int testBatchSize = 4;
    const int testClientNum = 4;
    //Test Env
	std::string testAddr = addrBase + std::to_string(testPort);
	LunarEnv testEnv(testAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "test env ready: " << testAddr);

	at::IntArrayRef testInputShape{testClientNum, 4};

	DqnOption testOption(testInputShape, deviceType);
	testOption.isAtari = false;
	testOption.donePerEp = 1;
	testOption.multiLifes = false;
	testOption.valueCoef = 0.25;

	testOption.toTest = true;
	testOption.inputScale = 1;
	testOption.batchSize = testBatchSize;
	testOption.testEp = testBatchSize;
	testOption.rewardScale = 1;
	testOption.rewardMin = -1;
	testOption.rewardMax = 1;
	testOption.tensorboardLogPath = logBase + "test" + "/" + logFileName;

	AlgTester<CartACFcNet, LunarEnv, SoftmaxPolicy> tester(model, testEnv, policy, testOption);

    while (true) {
    	sleep(pollMinute);

    	tester.test();
    }

    //Never reach
    for (int i = 0; i < workerNum; i ++) {
    	ts[i]->join();
    }
    updateThread->join();

    LOG4CXX_INFO(logger, "End of train");
}

//./testappo 50 4 32 16 10000000
void testpong0(const int clientNum, const int roundNum, const int maxStep, const int epochNum, const int updateNum) {
	//Env Config
	const int outputNum = 6;
	const std::string envName = "PongNoFrameskip-v4";


	//Training Config
    const int workerNum = roundNum;
    std::vector<float> entropyCoefs {0.01, 0.01, 0.005, 0.02, 0,02};
    assert(workerNum <= entropyCoefs.size());

    const int basePort = 10205;
    const int testPort = 10210;
    const std::string addrBase = "tcp://127.0.0.1:";
    const std::string logBase = "./logs/appo_testpong0/";
    const std::string logFileName = "tfevents.pb";

    //Net
    AirACHONet model(outputNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.0003));
    LOG4CXX_INFO(logger, "Model ready");
    SoftmaxPolicy policy(outputNum);
    std::mutex updateMutex;

	//Queue
	AsyncPPODataQ q(32);

	//Workers
	//TODO
	at::IntArrayRef inputShape{clientNum, 4, 84, 84};

    std::vector<AirEnv*> envs;
    std::vector<APPOWorker<AirACHONet, AirEnv, SoftmaxPolicy>*> workers;
    for(int i = 0; i < workerNum; i ++) {
    	DqnOption option(inputShape, deviceType);

    	option.isAtari = true;
    	option.donePerEp = 1;
    	option.multiLifes = false;
    	option.gamma = 0.99;
    	option.ppoLambda = 0.95;
    	option.inputScale = 255;
//    	option.batchSize = batchSize;
    	option.envNum = clientNum;
    	option.trajStepNum = maxStep;
    	option.rewardScale = 1;
    	option.rewardMin = -1;
    	option.rewardMax = 1;
    	option.tensorboardLogPath = logBase + std::to_string(i) + "/" + logFileName;
    	LOG4CXX_INFO(logger, "To put log in" << option.tensorboardLogPath);

//    	options.push_back(option);

    	std::string serverAddr = addrBase + std::to_string(basePort + i);
    	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
    	AirEnv* env = new AirEnv(serverAddr, envName, clientNum);
    	env->init();
    	envs.push_back(env);
    	LOG4CXX_INFO(logger, "Env " << envName << " " << i << " ready");

    	APPOWorker<AirACHONet, AirEnv, SoftmaxPolicy>* worker =
    			new APPOWorker<AirACHONet, AirEnv, SoftmaxPolicy> (
    					model,
						*env,
						policy,
						option,
						q,
						outputNum
    					);
    	workers.push_back(worker);
    }

	std::vector<std::unique_ptr<std::thread>> ts;

	for (int i = 0; i < workerNum; i ++) {
		ts.push_back(std::make_unique<std::thread>(
			&APPOWorker<AirACHONet, AirEnv, SoftmaxPolicy>::train, &(*workers[i]), updateNum));
	}

	////////////////////////////////////// Updater ////////////////////////////////////////
	at::IntArrayRef updateInputShape{clientNum, 4, 85, 84};

	DqnOption updateOption(updateInputShape, deviceType);
	updateOption.isAtari = true;
	updateOption.valueCoef = 0.25;
	updateOption.maxGradNormClip = 0.1;
	updateOption.gamma = 0.99;
	updateOption.ppoLambda = 0.95;
	updateOption.ppoEpsilon = 0.1;
	updateOption.appoRoundNum = roundNum;
	updateOption.epochNum = epochNum;
	updateOption.trajStepNum = maxStep * clientNum * roundNum;
//	updateOption.inputScale = 255;
//	updateOption.rewardScale = 1;
//	updateOption.rewardMin = -1;
//	updateOption.rewardMax = 1;
	updateOption.tensorboardLogPath = logBase + "update" + "/" + logFileName;

	APPOUpdater<AirACHONet, torch::optim::Adam> updater (
			model,
			optimizer,
			updateOption,
			q,
			outputNum
			);

	auto updateThread = std::make_unique<std::thread>(
			&APPOUpdater<AirACHONet, torch::optim::Adam>::train, &updater, updateNum);


	////////////////////////////////////// Test/////////////////////////////////////////
    const int pollMinute = 4 * 60;
    const int testBatchSize = 4;
    const int testClientNum = 4;
    //Test Env
	std::string testAddr = addrBase + std::to_string(testPort);
	AirEnv testEnv(testAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "test env ready: " << testAddr);

	at::IntArrayRef testInputShape{testClientNum, 4, 84, 84};

	DqnOption testOption(testInputShape, deviceType);
	testOption.donePerEp = 1;
	testOption.multiLifes = false;
	testOption.valueCoef = 0.25;

	testOption.toTest = true;
	testOption.inputScale = 255;
	testOption.batchSize = testBatchSize;
	testOption.testEp = testBatchSize;
	testOption.rewardScale = 1;
	testOption.rewardMin = -1;
	testOption.rewardMax = 1;
	testOption.tensorboardLogPath = logBase + "test" + "/" + logFileName;

	AlgTester<AirACHONet, AirEnv, SoftmaxPolicy> tester(model, testEnv, policy, testOption);

    while (true) {
    	sleep(pollMinute);

    	tester.test();
    }

    //Never reach
    for (int i = 0; i < workerNum; i ++) {
    	ts[i]->join();
    }
    updateThread->join();

    LOG4CXX_INFO(logger, "End of train");
}

//./testappo 50 4 32 16 10000000
//give up for error in previous version
void testpong1(const int clientNum, const int roundNum, const int maxStep, const int epochNum, const int updateNum) {
	//Env Config
	const int outputNum = 6;
	const std::string envName = "PongNoFrameskip-v4";


	//Training Config
    const int workerNum = roundNum;
//    std::vector<float> entropyCoefs {0.01, 0.01, 0.005, 0.02, 0,02};
//    assert(workerNum <= entropyCoefs.size());

    const int basePort = 10205;
    const int testPort = 10210;
    const std::string addrBase = "tcp://127.0.0.1:";
    const std::string logBase = "./logs/appo_testpong1/";
    const std::string logFileName = "tfevents.pb";

    //Net
    AirACHONet model(outputNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.0003));
    LOG4CXX_INFO(logger, "Model ready");
    SoftmaxPolicy policy(outputNum);
    std::mutex updateMutex;

	//Queue
	AsyncPPODataQ q(32);

	//Workers
	//TODO
	at::IntArrayRef inputShape{clientNum, 4, 84, 84};

    std::vector<AirEnv*> envs;
    std::vector<APPOWorker<AirACHONet, AirEnv, SoftmaxPolicy>*> workers;
    for(int i = 0; i < workerNum; i ++) {
    	DqnOption option(inputShape, deviceType);

    	option.isAtari = true;
    	option.donePerEp = 1;
    	option.multiLifes = false;
    	option.gamma = 0.99;
    	option.ppoLambda = 0.95;
    	option.inputScale = 255;
//    	option.batchSize = batchSize;
    	option.envNum = clientNum;
    	option.trajStepNum = maxStep;
    	option.rewardScale = 1;
    	option.rewardMin = -1;
    	option.rewardMax = 1;
    	option.tensorboardLogPath = logBase + std::to_string(i) + "/" + logFileName;
    	LOG4CXX_INFO(logger, "To put log in" << option.tensorboardLogPath);

//    	options.push_back(option);

    	std::string serverAddr = addrBase + std::to_string(basePort + i);
    	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
    	AirEnv* env = new AirEnv(serverAddr, envName, clientNum);
    	env->init();
    	envs.push_back(env);
    	LOG4CXX_INFO(logger, "Env " << envName << " " << i << " ready");

    	APPOWorker<AirACHONet, AirEnv, SoftmaxPolicy>* worker =
    			new APPOWorker<AirACHONet, AirEnv, SoftmaxPolicy> (
    					model,
						*env,
						policy,
						option,
						q,
						outputNum
    					);
    	workers.push_back(worker);
    }

	std::vector<std::unique_ptr<std::thread>> ts;

	for (int i = 0; i < workerNum; i ++) {
		ts.push_back(std::make_unique<std::thread>(
			&APPOWorker<AirACHONet, AirEnv, SoftmaxPolicy>::train, &(*workers[i]), updateNum));
	}

	////////////////////////////////////// Updater ////////////////////////////////////////
	at::IntArrayRef updateInputShape{clientNum, 4, 85, 84};

	DqnOption updateOption(updateInputShape, deviceType);
	updateOption.isAtari = true;
	updateOption.valueCoef = 0.25;
	updateOption.maxGradNormClip = 0.1;
	updateOption.gamma = 0.99;
	updateOption.ppoLambda = 0.95;
	updateOption.ppoEpsilon = 0.1;
	updateOption.appoRoundNum = roundNum;
	updateOption.epochNum = epochNum;
	updateOption.trajStepNum = maxStep * clientNum * roundNum;
	updateOption.entropyCoef = 0.05;
//	updateOption.inputScale = 255;
//	updateOption.rewardScale = 1;
//	updateOption.rewardMin = -1;
//	updateOption.rewardMax = 1;
	updateOption.logInterval = 4;
	updateOption.tensorboardLogPath = logBase + "update" + "/" + logFileName;

	APPOUpdater<AirACHONet, torch::optim::Adam> updater (
			model,
			optimizer,
			updateOption,
			q,
			outputNum
			);

	auto updateThread = std::make_unique<std::thread>(
			&APPOUpdater<AirACHONet, torch::optim::Adam>::train, &updater, updateNum);


	////////////////////////////////////// Test/////////////////////////////////////////
    const int pollMinute = 4 * 60;
    const int testBatchSize = 4;
    const int testClientNum = 4;
    //Test Env
	std::string testAddr = addrBase + std::to_string(testPort);
	AirEnv testEnv(testAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "test env ready: " << testAddr);

	at::IntArrayRef testInputShape{testClientNum, 4, 84, 84};

	DqnOption testOption(testInputShape, deviceType);
	testOption.donePerEp = 1;
	testOption.multiLifes = false;
//	testOption.valueCoef = 0.25;

	testOption.toTest = true;
	testOption.inputScale = 255;
	testOption.batchSize = testBatchSize;
	testOption.testEp = testBatchSize;
	testOption.rewardScale = 1;
	testOption.rewardMin = -1;
	testOption.rewardMax = 1;
	testOption.tensorboardLogPath = logBase + "test" + "/" + logFileName;

	AlgTester<AirACHONet, AirEnv, SoftmaxPolicy> tester(model, testEnv, policy, testOption);

    while (true) {
    	sleep(pollMinute);

    	tester.test();
    }

    //Never reach
    for (int i = 0; i < workerNum; i ++) {
    	ts[i]->join();
    }
    updateThread->join();

    LOG4CXX_INFO(logger, "End of train");
}

//./testappo 50 4 32 16 10000000
void testpong2(const int clientNum, const int roundNum, const int maxStep, const int epochNum, const int updateNum) {
	//Env Config
	const int outputNum = 6;
	const std::string envName = "PongNoFrameskip-v4";


	//Training Config
    const int workerNum = roundNum;
//    std::vector<float> entropyCoefs {0.01, 0.01, 0.005, 0.02, 0,02};
//    assert(workerNum <= entropyCoefs.size());

    const int basePort = 10205;
    const int testPort = 10210;
    const std::string addrBase = "tcp://127.0.0.1:";
    const std::string logBase = "./logs/appo_testpong2/";
    const std::string logFileName = "tfevents.pb";

    //Net
    AirACHONet model(outputNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.0003));
    LOG4CXX_INFO(logger, "Model ready");
    SoftmaxPolicy policy(outputNum);
    std::mutex updateMutex;

	//Queue
	AsyncPPODataQ q(32);

	//Workers
	//TODO
	at::IntArrayRef inputShape{clientNum, 4, 84, 84};

    std::vector<AirEnv*> envs;
    std::vector<APPOWorker<AirACHONet, AirEnv, SoftmaxPolicy>*> workers;
    for(int i = 0; i < workerNum; i ++) {
    	DqnOption option(inputShape, deviceType);

    	option.isAtari = true;
    	option.donePerEp = 1;
    	option.multiLifes = false;
    	option.gamma = 0.99;
    	option.ppoLambda = 0.95;
    	option.inputScale = 255;
//    	option.batchSize = batchSize;
    	option.envNum = clientNum;
    	option.trajStepNum = maxStep;
    	option.rewardScale = 1;
    	option.rewardMin = -1;
    	option.rewardMax = 1;
    	option.tensorboardLogPath = logBase + std::to_string(i) + "/" + logFileName;
    	LOG4CXX_INFO(logger, "To put log in" << option.tensorboardLogPath);

//    	options.push_back(option);

    	std::string serverAddr = addrBase + std::to_string(basePort + i);
    	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
    	AirEnv* env = new AirEnv(serverAddr, envName, clientNum);
    	env->init();
    	envs.push_back(env);
    	LOG4CXX_INFO(logger, "Env " << envName << " " << i << " ready");

    	APPOWorker<AirACHONet, AirEnv, SoftmaxPolicy>* worker =
    			new APPOWorker<AirACHONet, AirEnv, SoftmaxPolicy> (
    					model,
						*env,
						policy,
						option,
						q,
						outputNum
    					);
    	workers.push_back(worker);
    }

	std::vector<std::unique_ptr<std::thread>> ts;

	for (int i = 0; i < workerNum; i ++) {
		ts.push_back(std::make_unique<std::thread>(
			&APPOWorker<AirACHONet, AirEnv, SoftmaxPolicy>::train, &(*workers[i]), updateNum));
	}

	////////////////////////////////////// Updater ////////////////////////////////////////
	at::IntArrayRef updateInputShape{clientNum, 4, 85, 84};

	DqnOption updateOption(updateInputShape, deviceType);
	updateOption.isAtari = true;
	updateOption.valueCoef = 0.25;
	updateOption.maxGradNormClip = 0.1;
	updateOption.gamma = 0.99;
	updateOption.ppoLambda = 0.95;
	updateOption.ppoEpsilon = 0.1;
	updateOption.appoRoundNum = roundNum;
	updateOption.epochNum = epochNum;
	updateOption.trajStepNum = maxStep * clientNum * roundNum;
	updateOption.entropyCoef = 0.01;
//	updateOption.inputScale = 255;
//	updateOption.rewardScale = 1;
//	updateOption.rewardMin = -1;
//	updateOption.rewardMax = 1;
	updateOption.logInterval = 4;
	updateOption.tensorboardLogPath = logBase + "update" + "/" + logFileName;

	APPOUpdater<AirACHONet, torch::optim::Adam> updater (
			model,
			optimizer,
			updateOption,
			q,
			outputNum
			);

	auto updateThread = std::make_unique<std::thread>(
			&APPOUpdater<AirACHONet, torch::optim::Adam>::train, &updater, updateNum);


	////////////////////////////////////// Test/////////////////////////////////////////
    const int pollMinute = 4 * 60;
    const int testBatchSize = 4;
    const int testClientNum = 4;
    //Test Env
	std::string testAddr = addrBase + std::to_string(testPort);
	AirEnv testEnv(testAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "test env ready: " << testAddr);

	at::IntArrayRef testInputShape{testClientNum, 4, 84, 84};

	DqnOption testOption(testInputShape, deviceType);
	testOption.donePerEp = 1;
	testOption.multiLifes = false;
//	testOption.valueCoef = 0.25;

	testOption.toTest = true;
	testOption.inputScale = 255;
	testOption.batchSize = testBatchSize;
	testOption.testEp = testBatchSize;
	testOption.rewardScale = 1;
	testOption.rewardMin = -1;
	testOption.rewardMax = 1;
	testOption.tensorboardLogPath = logBase + "test" + "/" + logFileName;

	AlgTester<AirACHONet, AirEnv, SoftmaxPolicy> tester(model, testEnv, policy, testOption);

    while (true) {
    	sleep(pollMinute);

    	tester.test();
    }

    //Never reach
    for (int i = 0; i < workerNum; i ++) {
    	ts[i]->join();
    }
    updateThread->join();

    LOG4CXX_INFO(logger, "End of train");
}

//./testappo 50 4 32 16 10000000
void testbr3(const int clientNum, const int roundNum, const int maxStep, const int epochNum, const int updateNum) {
	//Env Config
	const int outputNum = 4;
	const std::string envName = "BreakoutNoFrameskip-v4";


	//Training Config
    const int workerNum = roundNum;
//    std::vector<float> entropyCoefs {0.01, 0.01, 0.005, 0.02, 0,02};
//    assert(workerNum <= entropyCoefs.size());

    const int basePort = 10205;
    const int testPort = 10210;
    const std::string addrBase = "tcp://127.0.0.1:";
    const std::string logBase = "./logs/appo_testbr4/";
    const std::string logFileName = "tfevents.pb";

    //Net
    AirACHONet model(outputNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.0003));
    LOG4CXX_INFO(logger, "Model ready");
    SoftmaxPolicy policy(outputNum);
//    std::mutex updateMutex;

	//Queue
	AsyncPPODataQ q(32);

	//Workers
	at::IntArrayRef inputShape{clientNum, 4, 84, 84};

    std::vector<AirEnv*> envs;
    std::vector<APPOWorker<AirACHONet, AirEnv, SoftmaxPolicy>*> workers;
    for(int i = 0; i < workerNum; i ++) {
    	DqnOption option(inputShape, deviceType);

    	option.isAtari = true;
    	option.gamma = 0.99;
    	option.ppoLambda = 0.95;
    	option.inputScale = 255;
//    	option.batchSize = batchSize;
    	option.envNum = clientNum;
    	option.trajStepNum = maxStep;
    	option.rewardScale = 1;
    	option.rewardMin = -1;
    	option.rewardMax = 1;
    	option.multiLifes = true;
    	option.donePerEp = 5;
    	option.tensorboardLogPath = logBase + std::to_string(i) + "/" + logFileName;
    	LOG4CXX_INFO(logger, "To put log in" << option.tensorboardLogPath);

//    	options.push_back(option);

    	std::string serverAddr = addrBase + std::to_string(basePort + i);
    	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
    	AirEnv* env = new AirEnv(serverAddr, envName, clientNum);
    	env->init();
    	envs.push_back(env);
    	LOG4CXX_INFO(logger, "Env " << envName << " " << i << " ready");

    	APPOWorker<AirACHONet, AirEnv, SoftmaxPolicy>* worker =
    			new APPOWorker<AirACHONet, AirEnv, SoftmaxPolicy> (
    					model,
						*env,
						policy,
						option,
						q,
						outputNum
    					);
    	workers.push_back(worker);
    }

	std::vector<std::unique_ptr<std::thread>> ts;

	for (int i = 0; i < workerNum; i ++) {
		ts.push_back(std::make_unique<std::thread>(
			&APPOWorker<AirACHONet, AirEnv, SoftmaxPolicy>::train, &(*workers[i]), updateNum));
	}

	////////////////////////////////////// Updater ////////////////////////////////////////
	at::IntArrayRef updateInputShape{clientNum, 4, 85, 84};

	DqnOption updateOption(updateInputShape, deviceType);
	updateOption.isAtari = true;
	updateOption.valueCoef = 0.25;
	updateOption.maxGradNormClip = 0.1;
	updateOption.gamma = 0.99;
	updateOption.ppoLambda = 0.95;
	updateOption.ppoEpsilon = 0.1;
	updateOption.appoRoundNum = roundNum;
	updateOption.epochNum = epochNum;
	updateOption.trajStepNum = maxStep * clientNum * roundNum;
	updateOption.entropyCoef = 0.01;
//	updateOption.inputScale = 255;
//	updateOption.rewardScale = 1;
//	updateOption.rewardMin = -1;
//	updateOption.rewardMax = 1;
	updateOption.logInterval = 4;
	updateOption.tensorboardLogPath = logBase + "update" + "/" + logFileName;

	APPOUpdater<AirACHONet, torch::optim::Adam> updater (
			model,
			optimizer,
			updateOption,
			q,
			outputNum
			);

	auto updateThread = std::make_unique<std::thread>(
			&APPOUpdater<AirACHONet, torch::optim::Adam>::train, &updater, updateNum);


	////////////////////////////////////// Test/////////////////////////////////////////
    const int pollMinute = 4 * 60;
    const int testBatchSize = 4;
    const int testClientNum = 4;
    //Test Env
	std::string testAddr = addrBase + std::to_string(testPort);
	AirEnv testEnv(testAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "test env ready: " << testAddr);

	at::IntArrayRef testInputShape{testClientNum, 4, 84, 84};

	DqnOption testOption(testInputShape, deviceType);
	testOption.donePerEp = 1;
	testOption.multiLifes = false;
//	testOption.valueCoef = 0.25;

	testOption.toTest = true;
	testOption.inputScale = 255;
	testOption.batchSize = testBatchSize;
	testOption.testEp = testBatchSize;
	testOption.rewardScale = 1;
	testOption.rewardMin = -1;
	testOption.rewardMax = 1;
	testOption.multiLifes = true;
	testOption.donePerEp = 5;
	testOption.tensorboardLogPath = logBase + "test" + "/" + logFileName;

	AlgTester<AirACHONet, AirEnv, SoftmaxPolicy> tester(model, testEnv, policy, testOption);

    while (true) {
    	sleep(pollMinute);

    	tester.test();
    }

    //Never reach
    for (int i = 0; i < workerNum; i ++) {
    	ts[i]->join();
    }
    updateThread->join();

    LOG4CXX_INFO(logger, "End of train");
}

//./testappo 50 5 32 20 10000000
void testqb4(const int clientNum, const int roundNum, const int maxStep, const int epochNum, const int updateNum) {
	//Env Config
	const int outputNum = 6;
	const std::string envName = "QbertNoFrameskip-v4";


	//Training Config
    const int workerNum = roundNum;
//    std::vector<float> entropyCoefs {0.01, 0.01, 0.005, 0.02, 0,02};
//    assert(workerNum <= entropyCoefs.size());

    const int basePort = 10205;
    const int testPort = 10210;
    const std::string addrBase = "tcp://127.0.0.1:";
    const std::string logBase = "./logs/appo_testqb4/";
    const std::string logFileName = "tfevents.pb";

    //Net
    AirACHONet model(outputNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.0003));
    LOG4CXX_INFO(logger, "Model ready");
    SoftmaxPolicy policy(outputNum);
//    std::mutex updateMutex;

	//Queue
	AsyncPPODataQ q(32);

	//Workers
	at::IntArrayRef inputShape{clientNum, 4, 84, 84};

    std::vector<AirEnv*> envs;
    std::vector<APPOWorker<AirACHONet, AirEnv, SoftmaxPolicy>*> workers;
    for(int i = 0; i < workerNum; i ++) {
    	DqnOption option(inputShape, deviceType);

    	option.isAtari = true;
    	option.gamma = 0.99;
    	option.ppoLambda = 0.95;
    	option.inputScale = 255;
//    	option.batchSize = batchSize;
    	option.envNum = clientNum;
    	option.trajStepNum = maxStep;
    	option.rewardScale = 1;
    	option.rewardMin = -1;
    	option.rewardMax = 1;
    	option.multiLifes = true;
    	option.donePerEp = 4;
    	option.tensorboardLogPath = logBase + std::to_string(i) + "/" + logFileName;
    	LOG4CXX_INFO(logger, "To put log in" << option.tensorboardLogPath);

//    	options.push_back(option);

    	std::string serverAddr = addrBase + std::to_string(basePort + i);
    	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
    	AirEnv* env = new AirEnv(serverAddr, envName, clientNum);
    	env->init();
    	envs.push_back(env);
    	LOG4CXX_INFO(logger, "Env " << envName << " " << i << " ready");

    	APPOWorker<AirACHONet, AirEnv, SoftmaxPolicy>* worker =
    			new APPOWorker<AirACHONet, AirEnv, SoftmaxPolicy> (
    					model,
						*env,
						policy,
						option,
						q,
						outputNum
    					);
    	workers.push_back(worker);
    }

	std::vector<std::unique_ptr<std::thread>> ts;

	for (int i = 0; i < workerNum; i ++) {
		ts.push_back(std::make_unique<std::thread>(
			&APPOWorker<AirACHONet, AirEnv, SoftmaxPolicy>::train, &(*workers[i]), updateNum));
	}

	////////////////////////////////////// Updater ////////////////////////////////////////
	at::IntArrayRef updateInputShape{clientNum, 4, 85, 84};

	DqnOption updateOption(updateInputShape, deviceType);
	updateOption.isAtari = true;
	updateOption.valueCoef = 0.5;
	updateOption.maxGradNormClip = 0.1;
	updateOption.gamma = 0.99;
	updateOption.ppoLambda = 0.95;
	updateOption.ppoEpsilon = 0.1;
	updateOption.appoRoundNum = roundNum;
	updateOption.epochNum = epochNum;
	updateOption.trajStepNum = maxStep * clientNum * roundNum;
	updateOption.entropyCoef = 0.01;
//	updateOption.inputScale = 255;
//	updateOption.rewardScale = 1;
//	updateOption.rewardMin = -1;
//	updateOption.rewardMax = 1;
	updateOption.logInterval = 4;
	updateOption.tensorboardLogPath = logBase + "update" + "/" + logFileName;

	APPOUpdater<AirACHONet, torch::optim::Adam> updater (
			model,
			optimizer,
			updateOption,
			q,
			outputNum
			);

	auto updateThread = std::make_unique<std::thread>(
			&APPOUpdater<AirACHONet, torch::optim::Adam>::train, &updater, updateNum);


	////////////////////////////////////// Test/////////////////////////////////////////
    const int pollMinute = 4 * 60;
    const int testBatchSize = 4;
    const int testClientNum = 4;
    //Test Env
	std::string testAddr = addrBase + std::to_string(testPort);
	AirEnv testEnv(testAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "test env ready: " << testAddr);

	at::IntArrayRef testInputShape{testClientNum, 4, 84, 84};

	DqnOption testOption(testInputShape, deviceType);
	testOption.donePerEp = 1;
	testOption.multiLifes = false;
//	testOption.valueCoef = 0.25;

	testOption.toTest = true;
	testOption.inputScale = 255;
	testOption.batchSize = testBatchSize;
	testOption.testEp = testBatchSize;
	testOption.rewardScale = 1;
	testOption.rewardMin = -1;
	testOption.rewardMax = 1;
	testOption.multiLifes = true;
	testOption.donePerEp = 4;
	testOption.tensorboardLogPath = logBase + "test" + "/" + logFileName;

	AlgTester<AirACHONet, AirEnv, SoftmaxPolicy> tester(model, testEnv, policy, testOption);

    while (true) {
    	sleep(pollMinute);

    	tester.test();
    }

    //Never reach
    for (int i = 0; i < workerNum; i ++) {
    	ts[i]->join();
    }
    updateThread->join();

    LOG4CXX_INFO(logger, "End of train");
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

	{
		int clienNum = atoi(argv[1]);
		int roundNum = atoi(argv[2]);
		int maxStep = atoi(argv[3]);
		int epochNum = atoi(argv[4]);
		int updateNum = atoi(argv[5]);
		testqb4(clienNum, roundNum, maxStep, epochNum, updateNum);
	}
	return 0;
}
