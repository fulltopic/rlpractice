/*
 * testappornn.cpp
 *
 *  Created on: Mar 6, 2022
 *      Author: zf
 */



#include "alg/rnn/appo/rnnappodataq.h"
#include "alg/rnn/appo/rnnappoupdater.hpp"
#include "alg/rnn/appo/rnnappoworker.hpp"
#include "alg/rnn/appo/rnnappocoupdater.hpp"
#include "alg/utils/algrnntester.hpp"

#include "gymtest/env/airenv.h"
#include "gymtest/env/lunarenv.h"
#include "gymtest/rnnnets/lunarnets/cartacgrutruncnet.h"
#include "gymtest/rnnnets/airnets/airacgrunet.h"
#include "gymtest/rnnnets/airnets/airacgrunordnet.h"


#include "gymtest/train/softmaxpolicy.h"
#include "alg/utils/dqnoption.h"


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

namespace {
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("testappornn"));
const torch::Device deviceType = torch::kCUDA;

void testCart(const int clientNum, const int batchSize, const int roundNum, const int maxStep, const int epochNum, const int updateNum) {
	//Env Config
	const int inputNum = 4;
	const int outputNum = 2;
	const int hiddenNum = 256;
	const int qCapacity = batchSize * roundNum * 2;
//	const int maxStep = 8;
	const std::string envName = "CartPole-v0";


	//Training Config
//	const int roundNum = 1;
//    const int maxStep = 5;
    const int workerNum = roundNum;
//    std::vector<float> entropyCoefs {0.01, 0.01, 0.005, 0.02, 0,02};
//    assert(workerNum <= entropyCoefs.size());

    const int basePort = 10201;
    const int testPort = 10210;
    const std::string addrBase = "tcp://127.0.0.1:";
    const std::string logBase = "./logs/appornn_testcart_r2e2m16/";
    const std::string logFileName = "tfevents.pb";

    //Net
    CartACGRUTruncFcNet model(inputNum, hiddenNum, outputNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(3e-4));
    LOG4CXX_INFO(logger, "Model ready");
    SoftmaxPolicy policy(outputNum);
    std::mutex updateMutex;

	//Queue
    AsyncRnnPPODataQ q(qCapacity);

	//Workers
	at::IntArrayRef inputShape{4};

    std::vector<LunarEnv*> envs;
    std::vector<RnnAPPOWorker<CartACGRUTruncFcNet, LunarEnv, SoftmaxPolicy>*> workers;
    for(int i = 0; i < workerNum; i ++) {
    	DqnOption option(inputShape, deviceType);

    	//env
    	option.envNum = clientNum;
    	option.isAtari = false;
    	option.donePerEp = 1;
    	option.multiLifes = false;
    	//grad
    	option.gamma = 0.99;
    	option.ppoLambda = 0.95;
    	//input
    	option.inputScale = 1;
    	option.rewardScale = 1;
    	option.rewardMin = -1;
    	option.rewardMax = 1;
    	//log
    	option.logInterval = 100;
    	option.tensorboardLogPath = logBase + std::to_string(i) + "/" + logFileName;
    	//model
    	option.saveModel = false;
    	option.loadModel = false;
    	//rnn
    	option.hiddenNums = {hiddenNum};
    	option.hidenLayerNums = {1};
    	option.gruCellNum = 1;
    	option.maxStep = 8;

    	LOG4CXX_INFO(logger, "To put log in" << option.tensorboardLogPath);


    	std::string serverAddr = addrBase + std::to_string(basePort + i);
    	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
    	LunarEnv* env = new LunarEnv(serverAddr, envName, clientNum);
    	env->init();
    	envs.push_back(env);
    	LOG4CXX_INFO(logger, "Env " << envName << " " << i << " ready");

    	RnnAPPOWorker<CartACGRUTruncFcNet, LunarEnv, SoftmaxPolicy>* worker =
    			new RnnAPPOWorker<CartACGRUTruncFcNet, LunarEnv, SoftmaxPolicy> (
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
			&RnnAPPOWorker<CartACGRUTruncFcNet, LunarEnv, SoftmaxPolicy>::train, &(*workers[i]), updateNum));
	}

	////////////////////////////////////// Updater ////////////////////////////////////////
	at::IntArrayRef updateInputShape{4};

	DqnOption updateOption(updateInputShape, deviceType);
	//env
	updateOption.isAtari = false;
	//grad
	updateOption.entropyCoef = 0.01;
	updateOption.valueCoef = 0.5;
	updateOption.maxGradNormClip = 0.5;
	updateOption.gamma = 0.99;
	//log
	updateOption.logInterval = 100;
	updateOption.tensorboardLogPath = logBase + "update" + "/" + logFileName;
	//input
	updateOption.inputScale = 1;
	updateOption.rewardScale = 1;
	updateOption.rewardMin = -1;
	updateOption.rewardMax = 1;
	//ppo
	updateOption.ppoLambda = 0.95;
	updateOption.ppoEpsilon = 0.1;
	updateOption.appoRoundNum = roundNum;
	updateOption.epochNum = epochNum;
	updateOption.batchSize = batchSize;
	updateOption.trajStepNum = batchSize * roundNum;
	updateOption.maxKl = 0.1;
	//test
	updateOption.toTest = false;
	//model
	updateOption.saveModel = false;
	updateOption.loadModel = false;
	//rnn
	updateOption.hiddenNums = {hiddenNum};
	updateOption.hidenLayerNums = {1};
	updateOption.gruCellNum = 1;


	RnnAPPOUpdater<CartACGRUTruncFcNet, torch::optim::Adam> updater (
			model,
			optimizer,
			updateOption,
			q,
			outputNum
			);

	auto updateThread = std::make_unique<std::thread>(
			&RnnAPPOUpdater<CartACGRUTruncFcNet, torch::optim::Adam>::train, &updater, updateNum);


	////////////////////////////////////// Test/////////////////////////////////////////
    const int pollMinute = 1 * 60;
    const int testBatchSize = 4;
    const int testClientNum = 4;
    //Test Env
	std::string testAddr = addrBase + std::to_string(testPort);
	LunarEnv testEnv(testAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "test env ready: " << testAddr);

	at::IntArrayRef testInputShape{4};

	DqnOption testOption(testInputShape, deviceType);
	testOption.isAtari = false;
	testOption.donePerEp = 1;
	testOption.multiLifes = false;
	testOption.valueCoef = 0.25;

	testOption.toTest = true;
	testOption.inputScale = 1;
	testOption.testBatch = testBatchSize;
	testOption.testEp = testBatchSize;
//	testOption.rewardScale = 1;
//	testOption.rewardMin = -1;
//	testOption.rewardMax = 1;
	testOption.tensorboardLogPath = logBase + "test" + "/" + logFileName;

	AlgRNNTester<CartACGRUTruncFcNet, LunarEnv, SoftmaxPolicy> tester(model, testEnv, policy, testOption);

    while (true) {
    	sleep(pollMinute);

    	tester.testAC();
    }

    //Never reach
    for (int i = 0; i < workerNum; i ++) {
    	ts[i]->join();
    }
    updateThread->join();

    LOG4CXX_INFO(logger, "End of train");
}

void testPong(const int clientNum, const int batchSize, const int roundNum, const int maxStep, const int epochNum, const int updateNum) {
	//Env Config
//	const int inputNum = 4;
	const int outputNum = 6;
	const int hiddenNum = 1024;
	const int qCapacity = batchSize * roundNum * 2;
//	const int maxStep = 8;
	const std::string envName = "PongNoFrameskip-v4";


	//Training Config
//	const int roundNum = 1;
//    const int maxStep = 5;
    const int workerNum = roundNum;
//    std::vector<float> entropyCoefs {0.01, 0.01, 0.005, 0.02, 0,02};
//    assert(workerNum <= entropyCoefs.size());

    const int basePort = 10201;
    const int testPort = 10210;
    const std::string addrBase = "tcp://127.0.0.1:";
    const std::string logBase = "./logs/appornn_testpong1_log/";
    const std::string logFileName = "tfevents.pb";

    //Net
    AirACHOGRUNet model(outputNum, hiddenNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(3e-4));
    LOG4CXX_INFO(logger, "Model ready");
    SoftmaxPolicy policy(outputNum);
    std::mutex updateMutex;

	//Queue
    AsyncRnnPPODataQ q(qCapacity);

	//Workers
	at::IntArrayRef inputShape{4, 84, 84};

    std::vector<AirEnv*> envs;
    std::vector<RnnAPPOWorker<AirACHOGRUNet, AirEnv, SoftmaxPolicy>*> workers;
    for(int i = 0; i < workerNum; i ++) {
    	DqnOption option(inputShape, deviceType);

    	//env
    	option.envNum = clientNum;
    	option.isAtari = true;
    	option.donePerEp = 1;
    	option.multiLifes = false;
    	//grad
    	option.gamma = 0.99;
    	option.ppoLambda = 0.95;
    	//input
    	option.inputScale = 255;
    	option.rewardScale = 1;
    	option.rewardMin = -1;
    	option.rewardMax = 1;
    	//log
    	option.logInterval = 100;
    	option.tensorboardLogPath = logBase + std::to_string(i) + "/" + logFileName;
    	//model
    	option.saveModel = false;
    	option.loadModel = false;
    	//rnn
    	option.hiddenNums = {hiddenNum};
    	option.hidenLayerNums = {1};
    	option.gruCellNum = 1;
    	option.maxStep = 8;

    	LOG4CXX_INFO(logger, "To put log in" << option.tensorboardLogPath);


    	std::string serverAddr = addrBase + std::to_string(basePort + i);
    	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
    	AirEnv* env = new AirEnv(serverAddr, envName, clientNum);
    	env->init();
    	envs.push_back(env);
    	LOG4CXX_INFO(logger, "Env " << envName << " " << i << " ready");

    	RnnAPPOWorker<AirACHOGRUNet, AirEnv, SoftmaxPolicy>* worker =
    			new RnnAPPOWorker<AirACHOGRUNet, AirEnv, SoftmaxPolicy> (
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
			&RnnAPPOWorker<AirACHOGRUNet, AirEnv, SoftmaxPolicy>::train, &(*workers[i]), updateNum));
	}

	////////////////////////////////////// Updater ////////////////////////////////////////
	at::IntArrayRef updateInputShape{4, 84, 84};

	DqnOption updateOption(updateInputShape, deviceType);
	//env
	updateOption.isAtari = false;
	//grad
	updateOption.entropyCoef = 0.01;
	updateOption.valueCoef = 0.5;
	updateOption.maxGradNormClip = 0.1;
	updateOption.gamma = 0.99;
	//log
	updateOption.logInterval = 100;
	updateOption.tensorboardLogPath = logBase + "update" + "/" + logFileName;
	//input
	updateOption.inputScale = 255;
	updateOption.rewardScale = 1;
	updateOption.rewardMin = -1;
	updateOption.rewardMax = 1;
	//ppo
	updateOption.ppoLambda = 0.95;
	updateOption.ppoEpsilon = 0.1;
	updateOption.appoRoundNum = roundNum;
	updateOption.epochNum = epochNum;
	updateOption.batchSize = batchSize;
	updateOption.trajStepNum = batchSize * roundNum;
	updateOption.maxKl = 0.1;
	//test
	updateOption.toTest = false;
	//model
	updateOption.saveModel = false;
	updateOption.loadModel = false;
	//rnn
	updateOption.hiddenNums = {hiddenNum};
	updateOption.hidenLayerNums = {1};
	updateOption.gruCellNum = 1;


	RnnAPPOUpdater<AirACHOGRUNet, torch::optim::Adam> updater (
			model,
			optimizer,
			updateOption,
			q,
			outputNum
			);

	auto updateThread = std::make_unique<std::thread>(
			&RnnAPPOUpdater<AirACHOGRUNet, torch::optim::Adam>::train, &updater, updateNum);


	////////////////////////////////////// Test/////////////////////////////////////////
    const int pollMinute = 4 * 60;
    const int testBatchSize = 4;
    const int testClientNum = 4;
    //Test Env
	std::string testAddr = addrBase + std::to_string(testPort);
	AirEnv testEnv(testAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "test env ready: " << testAddr);

	at::IntArrayRef testInputShape{4, 84, 84};

	DqnOption testOption(testInputShape, deviceType);
	testOption.isAtari = true;
	testOption.donePerEp = 1;
	testOption.multiLifes = false;
//	testOption.valueCoef = 0.25;

	testOption.toTest = true;
	testOption.inputScale = 255;
	testOption.testBatch = testBatchSize;
	testOption.testEp = testBatchSize;
//	testOption.rewardScale = 1;
//	testOption.rewardMin = -1;
//	testOption.rewardMax = 1;
	testOption.tensorboardLogPath = logBase + "test" + "/" + logFileName;

	AlgRNNTester<AirACHOGRUNet, AirEnv, SoftmaxPolicy> tester(model, testEnv, policy, testOption);

    while (true) {
    	sleep(pollMinute);

    	tester.testAC();
    }

    //Never reach
    for (int i = 0; i < workerNum; i ++) {
    	ts[i]->join();
    }
    updateThread->join();

    LOG4CXX_INFO(logger, "End of train");
}


void testPongNr(const int clientNum, const int batchSize, const int roundNum, const int maxStep, const int epochNum, const int updateNum) {
	//Env Config
//	const int inputNum = 4;
	const int outputNum = 6;
	const int hiddenNum = 1024;
	const int qCapacity = batchSize * roundNum * 2;
//	const int maxStep = 8;
	const std::string envName = "PongNoFrameskip-v4";


	//Training Config
//	const int roundNum = 1;
//    const int maxStep = 5;
    const int workerNum = roundNum;
//    std::vector<float> entropyCoefs {0.01, 0.01, 0.005, 0.02, 0,02};
//    assert(workerNum <= entropyCoefs.size());

    const int basePort = 10201;
    const int testPort = 10210;
    const std::string addrBase = "tcp://127.0.0.1:";
    const std::string logBase = "./logs/appornn_testpongnr_log/";
    const std::string logFileName = "tfevents.pb";

    //Net
    AirACHONRGRUNet model(outputNum, hiddenNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(3e-4));
    LOG4CXX_INFO(logger, "Model ready");
    SoftmaxPolicy policy(outputNum);
    std::mutex updateMutex;

	//Queue
    AsyncRnnPPODataQ q(qCapacity);

	//Workers
	at::IntArrayRef inputShape{4, 84, 84};

    std::vector<AirEnv*> envs;
    std::vector<RnnAPPOWorker<AirACHONRGRUNet, AirEnv, SoftmaxPolicy>*> workers;
    for(int i = 0; i < workerNum; i ++) {
    	DqnOption option(inputShape, deviceType);

    	//env
    	option.envNum = clientNum;
    	option.isAtari = true;
    	option.donePerEp = 1;
    	option.multiLifes = false;
    	//grad
    	option.gamma = 0.99;
    	option.ppoLambda = 0.95;
    	//input
    	option.inputScale = 255;
    	option.rewardScale = 1;
    	option.rewardMin = -1;
    	option.rewardMax = 1;
    	//log
    	option.logInterval = 100;
    	option.tensorboardLogPath = logBase + std::to_string(i) + "/" + logFileName;
    	//model
    	option.saveModel = false;
    	option.loadModel = false;
    	//rnn
    	option.hiddenNums = {hiddenNum};
    	option.hidenLayerNums = {1};
    	option.gruCellNum = 1;
    	option.maxStep = 8;

    	LOG4CXX_INFO(logger, "To put log in" << option.tensorboardLogPath);


    	std::string serverAddr = addrBase + std::to_string(basePort + i);
    	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
    	AirEnv* env = new AirEnv(serverAddr, envName, clientNum);
    	env->init();
    	envs.push_back(env);
    	LOG4CXX_INFO(logger, "Env " << envName << " " << i << " ready");

    	RnnAPPOWorker<AirACHONRGRUNet, AirEnv, SoftmaxPolicy>* worker =
    			new RnnAPPOWorker<AirACHONRGRUNet, AirEnv, SoftmaxPolicy> (
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
			&RnnAPPOWorker<AirACHONRGRUNet, AirEnv, SoftmaxPolicy>::train, &(*workers[i]), updateNum));
	}

	////////////////////////////////////// Updater ////////////////////////////////////////
	at::IntArrayRef updateInputShape{4, 84, 84};

	DqnOption updateOption(updateInputShape, deviceType);
	//env
	updateOption.isAtari = false;
	//grad
	updateOption.entropyCoef = 0.01;
	updateOption.valueCoef = 0.5;
	updateOption.maxGradNormClip = 0.1;
	updateOption.gamma = 0.99;
	//log
	updateOption.logInterval = 100;
	updateOption.tensorboardLogPath = logBase + "update" + "/" + logFileName;
	//input
	updateOption.inputScale = 255;
	updateOption.rewardScale = 1;
	updateOption.rewardMin = -1;
	updateOption.rewardMax = 1;
	//ppo
	updateOption.ppoLambda = 0.95;
	updateOption.ppoEpsilon = 0.1;
	updateOption.appoRoundNum = roundNum;
	updateOption.epochNum = epochNum;
	updateOption.batchSize = batchSize;
	updateOption.trajStepNum = batchSize * roundNum;
	updateOption.maxKl = 0.1;
	//test
	updateOption.toTest = false;
	//model
	updateOption.saveModel = false;
	updateOption.loadModel = false;
	//rnn
	updateOption.hiddenNums = {hiddenNum};
	updateOption.hidenLayerNums = {1};
	updateOption.gruCellNum = 1;


	RnnAPPOUpdater<AirACHONRGRUNet, torch::optim::Adam> updater (
			model,
			optimizer,
			updateOption,
			q,
			outputNum
			);

	auto updateThread = std::make_unique<std::thread>(
			&RnnAPPOUpdater<AirACHONRGRUNet, torch::optim::Adam>::train, &updater, updateNum);


	////////////////////////////////////// Test/////////////////////////////////////////
    const int pollMinute = 4 * 60;
    const int testBatchSize = 4;
    const int testClientNum = 4;
    //Test Env
	std::string testAddr = addrBase + std::to_string(testPort);
	AirEnv testEnv(testAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "test env ready: " << testAddr);

	at::IntArrayRef testInputShape{4, 84, 84};

	DqnOption testOption(testInputShape, deviceType);
	testOption.isAtari = true;
	testOption.donePerEp = 1;
	testOption.multiLifes = false;
//	testOption.valueCoef = 0.25;

	testOption.toTest = true;
	testOption.inputScale = 255;
	testOption.testBatch = testBatchSize;
	testOption.testEp = testBatchSize;
//	testOption.rewardScale = 1;
//	testOption.rewardMin = -1;
//	testOption.rewardMax = 1;
	testOption.tensorboardLogPath = logBase + "test" + "/" + logFileName;

	AlgRNNTester<AirACHONRGRUNet, AirEnv, SoftmaxPolicy> tester(model, testEnv, policy, testOption);

    while (true) {
    	sleep(pollMinute);

    	tester.testAC();
    }

    //Never reach
    for (int i = 0; i < workerNum; i ++) {
    	ts[i]->join();
    }
    updateThread->join();

    LOG4CXX_INFO(logger, "End of train");
}


void testPongNrAVec(const int clientNum, const int batchSize, const int roundNum, const int maxStep, const int epochNum, const int updateNum) {
	//Env Config
//	const int inputNum = 4;
	const int outputNum = 6;
	const int hiddenNum = 1024;
	const int qCapacity = batchSize * roundNum * 2;
//	const int maxStep = 8;
	const std::string envName = "PongNoFrameskip-v4";


	//Training Config
//	const int roundNum = 1;
//    const int maxStep = 5;
    const int workerNum = roundNum;
//    std::vector<float> entropyCoefs {0.01, 0.01, 0.005, 0.02, 0,02};
//    assert(workerNum <= entropyCoefs.size());

    const int basePort = 10201;
    const int testPort = 10210;
    const std::string addrBase = "tcp://127.0.0.1:";
    const std::string logBase = "./logs/appornn_testpongnravec_m16_log/";
    const std::string logFileName = "tfevents.pb";

    //Net
    AirACHONRGRUNet model(outputNum, hiddenNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(3e-4));
    LOG4CXX_INFO(logger, "Model ready");
    SoftmaxPolicy policy(outputNum);
    std::mutex updateMutex;

	//Queue
    AsyncRnnPPODataQ q(qCapacity);

	//Workers
	at::IntArrayRef inputShape{4, 84, 84};

    std::vector<AirEnv*> envs;
    std::vector<RnnAPPOWorker<AirACHONRGRUNet, AirEnv, SoftmaxPolicy>*> workers;
    for(int i = 0; i < workerNum; i ++) {
    	DqnOption option(inputShape, deviceType);

    	//env
    	option.envNum = clientNum;
    	option.isAtari = true;
    	option.donePerEp = 1;
    	option.multiLifes = false;
    	//grad
    	option.gamma = 0.99;
    	option.ppoLambda = 0.95;
    	//input
    	option.inputScale = 255;
    	option.rewardScale = 1;
    	option.rewardMin = -1;
    	option.rewardMax = 1;
    	//log
    	option.logInterval = 100;
    	option.tensorboardLogPath = logBase + std::to_string(i) + "/" + logFileName;
    	//model
    	option.saveModel = false;
    	option.loadModel = false;
    	//rnn
    	option.hiddenNums = {hiddenNum};
    	option.hidenLayerNums = {1};
    	option.gruCellNum = 1;
    	option.maxStep = 16;

    	LOG4CXX_INFO(logger, "To put log in" << option.tensorboardLogPath);


    	std::string serverAddr = addrBase + std::to_string(basePort + i);
    	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
    	AirEnv* env = new AirEnv(serverAddr, envName, clientNum);
    	env->init();
    	envs.push_back(env);
    	LOG4CXX_INFO(logger, "Env " << envName << " " << i << " ready");

    	RnnAPPOWorker<AirACHONRGRUNet, AirEnv, SoftmaxPolicy>* worker =
    			new RnnAPPOWorker<AirACHONRGRUNet, AirEnv, SoftmaxPolicy> (
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
			&RnnAPPOWorker<AirACHONRGRUNet, AirEnv, SoftmaxPolicy>::train, &(*workers[i]), updateNum));
	}

	////////////////////////////////////// Updater ////////////////////////////////////////
	at::IntArrayRef updateInputShape{4, 84, 84};

	DqnOption updateOption(updateInputShape, deviceType);
	//env
	updateOption.isAtari = false;
	//grad
	updateOption.entropyCoef = 0.01;
	updateOption.valueCoef = 0.5;
	updateOption.maxGradNormClip = 0.1;
	updateOption.gamma = 0.99;
	//log
	updateOption.logInterval = 100;
	updateOption.tensorboardLogPath = logBase + "update" + "/" + logFileName;
	//input
	updateOption.inputScale = 255;
	updateOption.rewardScale = 1;
	updateOption.rewardMin = -1;
	updateOption.rewardMax = 1;
	//ppo
	updateOption.ppoLambda = 0.95;
	updateOption.ppoEpsilon = 0.1;
	updateOption.appoRoundNum = roundNum;
	updateOption.epochNum = epochNum;
	updateOption.batchSize = batchSize;
	updateOption.trajStepNum = batchSize * roundNum;
	updateOption.maxKl = 0.1;
	//test
	updateOption.toTest = false;
	//model
	updateOption.saveModel = false;
	updateOption.loadModel = false;
	//rnn
	updateOption.hiddenNums = {hiddenNum};
	updateOption.hidenLayerNums = {1};
	updateOption.gruCellNum = 1;


	RnnAPPOUpdater<AirACHONRGRUNet, torch::optim::Adam> updater (
			model,
			optimizer,
			updateOption,
			q,
			outputNum
			);

	auto updateThread = std::make_unique<std::thread>(
			&RnnAPPOUpdater<AirACHONRGRUNet, torch::optim::Adam>::train, &updater, updateNum);


	////////////////////////////////////// Test/////////////////////////////////////////
    const int pollMinute = 4 * 60;
    const int testBatchSize = 4;
    const int testClientNum = 4;
    //Test Env
	std::string testAddr = addrBase + std::to_string(testPort);
	AirEnv testEnv(testAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "test env ready: " << testAddr);

	at::IntArrayRef testInputShape{4, 84, 84};

	DqnOption testOption(testInputShape, deviceType);
	testOption.isAtari = true;
	testOption.donePerEp = 1;
	testOption.multiLifes = false;
//	testOption.valueCoef = 0.25;

	testOption.toTest = true;
	testOption.inputScale = 255;
	testOption.testBatch = testBatchSize;
	testOption.testEp = testBatchSize;
//	testOption.rewardScale = 1;
//	testOption.rewardMin = -1;
//	testOption.rewardMax = 1;
	testOption.tensorboardLogPath = logBase + "test" + "/" + logFileName;

	AlgRNNTester<AirACHONRGRUNet, AirEnv, SoftmaxPolicy> tester(model, testEnv, policy, testOption);

    while (true) {
    	sleep(pollMinute);

    	tester.testAC();
    }

    //Never reach
    for (int i = 0; i < workerNum; i ++) {
    	ts[i]->join();
    }
    updateThread->join();

    LOG4CXX_INFO(logger, "End of train");
}


void testBrNr(const int clientNum, const int batchSize, const int roundNum, const int maxStep, const int epochNum, const int updateNum) {
	//Env Config
//	const int inputNum = 4;
	const int outputNum = 4;
	const int hiddenNum = 1024;
	const int qCapacity = batchSize * roundNum * 2;
//	const int maxStep = 8;
	const std::string envName = "BreakoutNoFrameskip-v4";


	//Training Config
//	const int roundNum = 1;
//    const int maxStep = 5;
    const int workerNum = roundNum;
//    std::vector<float> entropyCoefs {0.01, 0.01, 0.005, 0.02, 0,02};
//    assert(workerNum <= entropyCoefs.size());

    const int basePort = 10201;
    const int testPort = 10210;
    const std::string addrBase = "tcp://127.0.0.1:";
    const std::string logBase = "./logs/appornn_testbrnr_log/";
    const std::string logFileName = "tfevents.pb";

    //Net
    AirACHONRGRUNet model(outputNum, hiddenNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(3e-4));
    LOG4CXX_INFO(logger, "Model ready");
    SoftmaxPolicy policy(outputNum);
//    std::mutex updateMutex;

	//Queue
    AsyncRnnPPODataQ q(qCapacity);

	//Workers
	at::IntArrayRef inputShape{4, 84, 84};

    std::vector<AirEnv*> envs;
    std::vector<RnnAPPOWorker<AirACHONRGRUNet, AirEnv, SoftmaxPolicy>*> workers;
    for(int i = 0; i < workerNum; i ++) {
    	DqnOption option(inputShape, deviceType);

    	//env
    	option.envNum = clientNum;
    	option.isAtari = true;
    	option.donePerEp = 5;
    	option.multiLifes = true;
    	//grad
    	option.gamma = 0.99;
    	option.ppoLambda = 0.95;
    	//input
    	option.inputScale = 255;
    	option.rewardScale = 1;
    	option.rewardMin = -1;
    	option.rewardMax = 1;
    	//log
    	option.logInterval = 100;
    	option.tensorboardLogPath = logBase + std::to_string(i) + "/" + logFileName;
    	//model
    	option.saveModel = false;
    	option.loadModel = false;
    	//rnn
    	option.hiddenNums = {hiddenNum};
    	option.hidenLayerNums = {1};
    	option.gruCellNum = 1;
    	option.maxStep = 8;

    	LOG4CXX_INFO(logger, "To put log in" << option.tensorboardLogPath);


    	std::string serverAddr = addrBase + std::to_string(basePort + i);
    	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
    	AirEnv* env = new AirEnv(serverAddr, envName, clientNum);
    	env->init();
    	envs.push_back(env);
    	LOG4CXX_INFO(logger, "Env " << envName << " " << i << " ready");

    	RnnAPPOWorker<AirACHONRGRUNet, AirEnv, SoftmaxPolicy>* worker =
    			new RnnAPPOWorker<AirACHONRGRUNet, AirEnv, SoftmaxPolicy> (
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
			&RnnAPPOWorker<AirACHONRGRUNet, AirEnv, SoftmaxPolicy>::train, &(*workers[i]), updateNum));
	}

	////////////////////////////////////// Updater ////////////////////////////////////////
	at::IntArrayRef updateInputShape{4, 84, 84};

	DqnOption updateOption(updateInputShape, deviceType);
	//env
	updateOption.isAtari = true;
	//grad
	updateOption.entropyCoef = 0.01;
	updateOption.valueCoef = 0.5;
	updateOption.maxGradNormClip = 0.1;
	updateOption.gamma = 0.99;
	//log
	updateOption.logInterval = 100;
	updateOption.tensorboardLogPath = logBase + "update" + "/" + logFileName;
	//input
	updateOption.inputScale = 255;
	updateOption.rewardScale = 1;
	updateOption.rewardMin = -1;
	updateOption.rewardMax = 1;
	//ppo
	updateOption.ppoLambda = 0.95;
	updateOption.ppoEpsilon = 0.1;
	updateOption.appoRoundNum = roundNum;
	updateOption.epochNum = epochNum;
	updateOption.batchSize = batchSize;
	updateOption.trajStepNum = batchSize * roundNum;
	updateOption.maxKl = 0.1;
	//test
	updateOption.toTest = false;
	//model
	updateOption.saveModel = false;
	updateOption.loadModel = false;
	//rnn
	updateOption.hiddenNums = {hiddenNum};
	updateOption.hidenLayerNums = {1};
	updateOption.gruCellNum = 1;


	RnnAPPOUpdater<AirACHONRGRUNet, torch::optim::Adam> updater (
			model,
			optimizer,
			updateOption,
			q,
			outputNum
			);

	auto updateThread = std::make_unique<std::thread>(
			&RnnAPPOUpdater<AirACHONRGRUNet, torch::optim::Adam>::train, &updater, updateNum);


	////////////////////////////////////// Test/////////////////////////////////////////
    const int pollMinute = 4 * 60;
    const int testBatchSize = 4;
    const int testClientNum = 4;
    //Test Env
	std::string testAddr = addrBase + std::to_string(testPort);
	AirEnv testEnv(testAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "test env ready: " << testAddr);

	at::IntArrayRef testInputShape{4, 84, 84};

	DqnOption testOption(testInputShape, deviceType);
	testOption.isAtari = true;
	testOption.donePerEp = 5;
	testOption.multiLifes = true;
//	testOption.valueCoef = 0.25;

	testOption.toTest = true;
	testOption.inputScale = 255;
	testOption.testBatch = testBatchSize;
	testOption.testEp = testBatchSize;
//	testOption.rewardScale = 1;
//	testOption.rewardMin = -1;
//	testOption.rewardMax = 1;
	testOption.tensorboardLogPath = logBase + "test" + "/" + logFileName;

	AlgRNNTester<AirACHONRGRUNet, AirEnv, SoftmaxPolicy> tester(model, testEnv, policy, testOption);

    while (true) {
    	sleep(pollMinute);

    	tester.testAC();
    }

    //Never reach
    for (int i = 0; i < workerNum; i ++) {
    	ts[i]->join();
    }
    updateThread->join();

    LOG4CXX_INFO(logger, "End of train");
}


void testBrNrAVec(const int clientNum, const int batchSize, const int roundNum, const int maxStep, const int epochNum, const int updateNum) {
	//Env Config
//	const int inputNum = 4;
	const int outputNum = 4;
	const int hiddenNum = 1024;
	const int qCapacity = batchSize * roundNum * 2;
//	const int maxStep = 8;
	const std::string envName = "BreakoutNoFrameskip-v4";


	//Training Config
//	const int roundNum = 1;
//    const int maxStep = 5;
    const int workerNum = roundNum;
//    std::vector<float> entropyCoefs {0.01, 0.01, 0.005, 0.02, 0,02};
//    assert(workerNum <= entropyCoefs.size());

    const int basePort = 10201;
    const int testPort = 10210;
    const std::string addrBase = "tcp://127.0.0.1:";
    const std::string logBase = "./logs/appornn_testbrnravec_m16_log/";
    const std::string logFileName = "tfevents.pb";

    //Net
    AirACHONRGRUNet model(outputNum, hiddenNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
    LOG4CXX_INFO(logger, "Model ready");
    SoftmaxPolicy policy(outputNum);
//    std::mutex updateMutex;

	//Queue
    AsyncRnnPPODataQ q(qCapacity);

	//Workers
	at::IntArrayRef inputShape{4, 84, 84};

    std::vector<AirEnv*> envs;
    std::vector<RnnAPPOWorker<AirACHONRGRUNet, AirEnv, SoftmaxPolicy>*> workers;
    for(int i = 0; i < workerNum; i ++) {
    	DqnOption option(inputShape, deviceType);

    	//env
    	option.envNum = clientNum;
    	option.isAtari = true;
    	option.donePerEp = 5;
    	option.multiLifes = true;
    	//grad
    	option.gamma = 0.99;
    	option.ppoLambda = 0.95;
    	//input
    	option.inputScale = 255;
    	option.rewardScale = 1;
    	option.rewardMin = -1;
    	option.rewardMax = 1;
    	//log
    	option.logInterval = 100;
    	option.tensorboardLogPath = logBase + std::to_string(i) + "/" + logFileName;
    	//model
    	option.saveModel = false;
    	option.loadModel = false;
    	//rnn
    	option.hiddenNums = {hiddenNum};
    	option.hidenLayerNums = {1};
    	option.gruCellNum = 1;
    	option.maxStep = 16;

    	LOG4CXX_INFO(logger, "To put log in" << option.tensorboardLogPath);


    	std::string serverAddr = addrBase + std::to_string(basePort + i);
    	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
    	AirEnv* env = new AirEnv(serverAddr, envName, clientNum);
    	env->init();
    	envs.push_back(env);
    	LOG4CXX_INFO(logger, "Env " << envName << " " << i << " ready");

    	RnnAPPOWorker<AirACHONRGRUNet, AirEnv, SoftmaxPolicy>* worker =
    			new RnnAPPOWorker<AirACHONRGRUNet, AirEnv, SoftmaxPolicy> (
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
			&RnnAPPOWorker<AirACHONRGRUNet, AirEnv, SoftmaxPolicy>::train, &(*workers[i]), updateNum));
	}

	////////////////////////////////////// Updater ////////////////////////////////////////
	at::IntArrayRef updateInputShape{4, 84, 84};

	DqnOption updateOption(updateInputShape, deviceType);
	//env
	updateOption.isAtari = true;
	//grad
	updateOption.entropyCoef = 0.01;
	updateOption.valueCoef = 0.5;
	updateOption.maxGradNormClip = 0.1;
	updateOption.gamma = 0.99;
	//log
	updateOption.logInterval = 100;
	updateOption.tensorboardLogPath = logBase + "update" + "/" + logFileName;
	//input
	updateOption.inputScale = 255;
	updateOption.rewardScale = 1;
	updateOption.rewardMin = -1;
	updateOption.rewardMax = 1;
	//ppo
	updateOption.ppoLambda = 0.95;
	updateOption.ppoEpsilon = 0.1;
	updateOption.appoRoundNum = roundNum;
	updateOption.epochNum = epochNum;
	updateOption.batchSize = batchSize;
	updateOption.trajStepNum = batchSize * roundNum;
	updateOption.maxKl = 0.1;
	//test
	updateOption.toTest = false;
	//model
	updateOption.saveModel = false;
	updateOption.loadModel = false;
	//rnn
	updateOption.hiddenNums = {hiddenNum};
	updateOption.hidenLayerNums = {1};
	updateOption.gruCellNum = 1;


	RnnAPPOUpdater<AirACHONRGRUNet, torch::optim::Adam> updater (
			model,
			optimizer,
			updateOption,
			q,
			outputNum
			);

	auto updateThread = std::make_unique<std::thread>(
			&RnnAPPOUpdater<AirACHONRGRUNet, torch::optim::Adam>::train, &updater, updateNum);


	////////////////////////////////////// Test/////////////////////////////////////////
    const int pollMinute = 4 * 60;
    const int testBatchSize = 4;
    const int testClientNum = 4;
    //Test Env
	std::string testAddr = addrBase + std::to_string(testPort);
	AirEnv testEnv(testAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "test env ready: " << testAddr);

	at::IntArrayRef testInputShape{4, 84, 84};

	DqnOption testOption(testInputShape, deviceType);
	testOption.isAtari = true;
	testOption.donePerEp = 5;
	testOption.multiLifes = true;
//	testOption.valueCoef = 0.25;

	testOption.toTest = true;
	testOption.inputScale = 255;
	testOption.testBatch = testBatchSize;
	testOption.testEp = testBatchSize;
//	testOption.rewardScale = 1;
//	testOption.rewardMin = -1;
//	testOption.rewardMax = 1;
	testOption.tensorboardLogPath = logBase + "test" + "/" + logFileName;

	AlgRNNTester<AirACHONRGRUNet, AirEnv, SoftmaxPolicy> tester(model, testEnv, policy, testOption);

    while (true) {
    	sleep(pollMinute);

    	tester.testAC();
    }

    //Never reach
    for (int i = 0; i < workerNum; i ++) {
    	ts[i]->join();
    }
    updateThread->join();

    LOG4CXX_INFO(logger, "End of train");
}

void testBrNr1(const int clientNum, const int batchSize, const int roundNum, const int maxStep, const int epochNum, const int updateNum) {
	//Env Config
//	const int inputNum = 4;
	const int outputNum = 4;
	const int hiddenNum = 1024;
	const int qCapacity = batchSize * roundNum * 2;
//	const int maxStep = 8;
	const std::string envName = "BreakoutNoFrameskip-v4";


	//Training Config
//	const int roundNum = 1;
//    const int maxStep = 5;
    const int workerNum = roundNum;
//    std::vector<float> entropyCoefs {0.01, 0.01, 0.005, 0.02, 0,02};
//    assert(workerNum <= entropyCoefs.size());

    const int basePort = 10201;
    const int testPort = 10210;
    const std::string addrBase = "tcp://127.0.0.1:";
    const std::string logBase = "./logs/appornn_testbrnr1_log/";
    const std::string logFileName = "tfevents.pb";

    //Net
    AirACHONRGRUNet model(outputNum, hiddenNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
    LOG4CXX_INFO(logger, "Model ready");
    SoftmaxPolicy policy(outputNum);
//    std::mutex updateMutex;

	//Queue
    AsyncRnnPPODataQ q(qCapacity);

	//Workers
	at::IntArrayRef inputShape{4, 84, 84};

    std::vector<AirEnv*> envs;
    std::vector<RnnAPPOWorker<AirACHONRGRUNet, AirEnv, SoftmaxPolicy>*> workers;
    for(int i = 0; i < workerNum; i ++) {
    	DqnOption option(inputShape, deviceType);

    	//env
    	option.envNum = clientNum;
    	option.isAtari = true;
    	option.donePerEp = 5;
    	option.multiLifes = true;
    	//grad
    	option.gamma = 0.99;
    	option.ppoLambda = 0.95;
    	//input
    	option.inputScale = 255;
    	option.rewardScale = 1;
    	option.rewardMin = -1;
    	option.rewardMax = 1;
    	//log
    	option.logInterval = 100;
    	option.tensorboardLogPath = logBase + std::to_string(i) + "/" + logFileName;
    	//model
    	option.saveModel = false;
    	option.loadModel = false;
    	//rnn
    	option.hiddenNums = {hiddenNum};
    	option.hidenLayerNums = {1};
    	option.gruCellNum = 1;
    	option.maxStep = 8;

    	LOG4CXX_INFO(logger, "To put log in" << option.tensorboardLogPath);


    	std::string serverAddr = addrBase + std::to_string(basePort + i);
    	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
    	AirEnv* env = new AirEnv(serverAddr, envName, clientNum);
    	env->init();
    	envs.push_back(env);
    	LOG4CXX_INFO(logger, "Env " << envName << " " << i << " ready");

    	RnnAPPOWorker<AirACHONRGRUNet, AirEnv, SoftmaxPolicy>* worker =
    			new RnnAPPOWorker<AirACHONRGRUNet, AirEnv, SoftmaxPolicy> (
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
			&RnnAPPOWorker<AirACHONRGRUNet, AirEnv, SoftmaxPolicy>::train, &(*workers[i]), updateNum));
	}

	////////////////////////////////////// Updater ////////////////////////////////////////
	at::IntArrayRef updateInputShape{4, 84, 84};

	DqnOption updateOption(updateInputShape, deviceType);
	//env
	updateOption.isAtari = true;
	//grad
	updateOption.entropyCoef = 0.01;
	updateOption.valueCoef = 0.5;
	updateOption.maxGradNormClip = 0.1;
	updateOption.gamma = 0.99;
	//log
	updateOption.logInterval = 100;
	updateOption.tensorboardLogPath = logBase + "update" + "/" + logFileName;
	//input
	updateOption.inputScale = 255;
	updateOption.rewardScale = 1;
	updateOption.rewardMin = -1;
	updateOption.rewardMax = 1;
	//ppo
	updateOption.ppoLambda = 0.95;
	updateOption.ppoEpsilon = 0.1;
	updateOption.appoRoundNum = roundNum;
	updateOption.epochNum = epochNum;
	updateOption.batchSize = batchSize;
	updateOption.trajStepNum = batchSize * roundNum;
	updateOption.maxKl = 0.1;
	//test
	updateOption.toTest = false;
	//model
	updateOption.saveModel = false;
	updateOption.loadModel = false;
	//rnn
	updateOption.hiddenNums = {hiddenNum};
	updateOption.hidenLayerNums = {1};
	updateOption.gruCellNum = 1;


	RnnAPPOUpdater<AirACHONRGRUNet, torch::optim::Adam> updater (
			model,
			optimizer,
			updateOption,
			q,
			outputNum
			);

	auto updateThread = std::make_unique<std::thread>(
			&RnnAPPOUpdater<AirACHONRGRUNet, torch::optim::Adam>::train, &updater, updateNum);


	////////////////////////////////////// Test/////////////////////////////////////////
    const int pollMinute = 4 * 60;
    const int testBatchSize = 4;
    const int testClientNum = 4;
    //Test Env
	std::string testAddr = addrBase + std::to_string(testPort);
	AirEnv testEnv(testAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "test env ready: " << testAddr);

	at::IntArrayRef testInputShape{4, 84, 84};

	DqnOption testOption(testInputShape, deviceType);
	testOption.isAtari = true;
	testOption.donePerEp = 5;
	testOption.multiLifes = true;
//	testOption.valueCoef = 0.25;

	testOption.toTest = true;
	testOption.inputScale = 255;
	testOption.testBatch = testBatchSize;
	testOption.testEp = testBatchSize;
//	testOption.rewardScale = 1;
//	testOption.rewardMin = -1;
//	testOption.rewardMax = 1;
	testOption.tensorboardLogPath = logBase + "test" + "/" + logFileName;

	AlgRNNTester<AirACHONRGRUNet, AirEnv, SoftmaxPolicy> tester(model, testEnv, policy, testOption);

    while (true) {
    	sleep(pollMinute);

    	tester.testAC();
    }

    //Never reach
    for (int i = 0; i < workerNum; i ++) {
    	ts[i]->join();
    }
    updateThread->join();

    LOG4CXX_INFO(logger, "End of train");
}


void testCoPong(const int clientNum, const int updaterNum,
		const int batchSize, const int roundNum, const int maxStep, const int epochNum, const int updateNum) {
	//Env Config
	const int outputNum = 6;
	const int hiddenNum = 1024;
	const int qCapacity = batchSize * roundNum * 2;
	const std::string envName = "PongNoFrameskip-v4";


    const int workerNum = roundNum;
//    std::vector<float> entropyCoefs {0.01, 0.01, 0.005, 0.02, 0,02};
//    assert(workerNum <= entropyCoefs.size());

    const int basePort = 10201;
    const int testPort = 10210;
    const std::string addrBase = "tcp://127.0.0.1:";
    const std::string logBase = "./logs/appornn_testpong_log/";
    const std::string logFileName = "tfevents.pb";

    //Net
    AirACHOGRUNet model(outputNum, hiddenNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(3e-4));
    LOG4CXX_INFO(logger, "Model ready");
    SoftmaxPolicy policy(outputNum);
    std::mutex updateMutex;

	//Queue
    AsyncRnnPPODataQ q(qCapacity);

	//Workers
	at::IntArrayRef inputShape{4, 84, 84};

    std::vector<AirEnv*> envs;
    std::vector<RnnAPPOWorker<AirACHOGRUNet, AirEnv, SoftmaxPolicy>*> workers;
    for(int i = 0; i < workerNum; i ++) {
    	DqnOption option(inputShape, deviceType);

    	//env
    	option.envNum = clientNum;
    	option.isAtari = true;
    	option.donePerEp = 1;
    	option.multiLifes = false;
    	//grad
    	option.gamma = 0.99;
    	option.ppoLambda = 0.95;
    	//input
    	option.inputScale = 255;
    	option.rewardScale = 1;
    	option.rewardMin = -1;
    	option.rewardMax = 1;
    	//log
    	option.logInterval = 100;
    	option.tensorboardLogPath = logBase + std::to_string(i) + "/" + logFileName;
    	//model
    	option.saveModel = false;
    	option.loadModel = false;
    	//rnn
    	option.hiddenNums = {hiddenNum};
    	option.hidenLayerNums = {1};
    	option.gruCellNum = 1;
    	option.maxStep = 8;

    	LOG4CXX_INFO(logger, "To put log in" << option.tensorboardLogPath);


    	std::string serverAddr = addrBase + std::to_string(basePort + i);
    	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
    	AirEnv* env = new AirEnv(serverAddr, envName, clientNum);
    	env->init();
    	envs.push_back(env);
    	LOG4CXX_INFO(logger, "Env " << envName << " " << i << " ready");

    	RnnAPPOWorker<AirACHOGRUNet, AirEnv, SoftmaxPolicy>* worker =
    			new RnnAPPOWorker<AirACHOGRUNet, AirEnv, SoftmaxPolicy> (
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
			&RnnAPPOWorker<AirACHOGRUNet, AirEnv, SoftmaxPolicy>::train, &(*workers[i]), updateNum));
	}

	////////////////////////////////////// Updater ////////////////////////////////////////
	at::IntArrayRef updateInputShape{4, 84, 84};

	DqnOption updateOption(updateInputShape, deviceType);
	//env
	updateOption.isAtari = false;
	//grad
	updateOption.entropyCoef = 0.01;
	updateOption.valueCoef = 0.5;
	updateOption.maxGradNormClip = 0.1;
	updateOption.gamma = 0.99;
	//log
	updateOption.logInterval = 100;
	updateOption.tensorboardLogPath = logBase + "update" + "/" + logFileName;
	//input
	updateOption.inputScale = 255;
	updateOption.rewardScale = 1;
	updateOption.rewardMin = -1;
	updateOption.rewardMax = 1;
	//ppo
	updateOption.ppoLambda = 0.95;
	updateOption.ppoEpsilon = 0.1;
	updateOption.appoRoundNum = roundNum;
	updateOption.epochNum = epochNum;
	updateOption.batchSize = batchSize;
	updateOption.trajStepNum = batchSize * roundNum;
	updateOption.maxKl = 0.1;
	//test
	updateOption.toTest = false;
	//model
	updateOption.saveModel = false;
	updateOption.loadModel = false;
	//rnn
	updateOption.hiddenNums = {hiddenNum};
	updateOption.hidenLayerNums = {1};
	updateOption.gruCellNum = 1;

	std::vector<RnnAPPOCoUpdater<AirACHOGRUNet, torch::optim::Adam>*> updaters;
	for (int i = 0; i < updaterNum; i ++) {
		updateOption.tensorboardLogPath = logBase + "update" + std::to_string(i) + "/" + logFileName;

		RnnAPPOCoUpdater<AirACHOGRUNet, torch::optim::Adam>* updater =
    			new RnnAPPOCoUpdater<AirACHOGRUNet, torch::optim::Adam> (
    					model,
						optimizer,
						updateOption,
						q,
						updateMutex,
						outputNum
    					);

    	updaters.push_back(updater);
	}
	for (int i = 0; i < updaterNum; i ++) {
		ts.push_back(std::make_unique<std::thread>(
			&RnnAPPOCoUpdater<AirACHOGRUNet, torch::optim::Adam>::train, &(*updaters[i]), updateNum));
	}

//	RnnAPPOUpdater<AirACHOGRUNet, torch::optim::Adam> updater (
//			model,
//			optimizer,
//			updateOption,
//			q,
//			outputNum
//			);
//
//	auto updateThread = std::make_unique<std::thread>(
//			&RnnAPPOUpdater<AirACHOGRUNet, torch::optim::Adam>::train, &updater, updateNum);


	////////////////////////////////////// Test/////////////////////////////////////////
    const int pollMinute = 4 * 60;
    const int testBatchSize = 4;
    const int testClientNum = 4;
    //Test Env
	std::string testAddr = addrBase + std::to_string(testPort);
	AirEnv testEnv(testAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "test env ready: " << testAddr);

	at::IntArrayRef testInputShape{4, 84, 84};

	DqnOption testOption(testInputShape, deviceType);
	testOption.isAtari = true;
	testOption.donePerEp = 1;
	testOption.multiLifes = false;
//	testOption.valueCoef = 0.25;

	testOption.toTest = true;
	testOption.inputScale = 255;
	testOption.testBatch = testBatchSize;
	testOption.testEp = testBatchSize;
//	testOption.rewardScale = 1;
//	testOption.rewardMin = -1;
//	testOption.rewardMax = 1;
	testOption.tensorboardLogPath = logBase + "test" + "/" + logFileName;

	AlgRNNTester<AirACHOGRUNet, AirEnv, SoftmaxPolicy> tester(model, testEnv, policy, testOption);

    while (true) {
    	sleep(pollMinute);

    	tester.testAC();
    }

    //Never reach
    for (int i = 0; i < workerNum; i ++) {
    	ts[i]->join();
    }
//    updateThread->join();

//    LOG4CXX_INFO(logger, "End of train");
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
		int clientNum = atoi(argv[1]);
		int batchNum = atoi(argv[2]);
		int roundNum = atoi(argv[3]);
		int maxStep = atoi(argv[4]);
		int epochNum = atoi(argv[5]);
		int updateNum = atoi(argv[6]);
//		testCart(clientNum, batchNum, roundNum, maxStep, epochNum, updateNum);
//		testPongNrAVec(clientNum, batchNum, roundNum, maxStep, epochNum, updateNum);
		testBrNrAVec(clientNum, batchNum, roundNum, maxStep, epochNum, updateNum);
//		testCoPong(clientNum, roundNum, batchNum, roundNum, maxStep, epochNum, updateNum);
	}
	return 0;
}
