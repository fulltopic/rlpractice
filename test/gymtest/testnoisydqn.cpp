/*
 * testnoisydqn.cpp
 *
 *  Created on: Aug 31, 2021
 *      Author: zf
 */





#include "alg/noisydoubledqn.hpp"
#include "alg/noisydqn.hpp"

#include "gymtest/env/airenv.h"
#include "gymtest/env/lunarenv.h"
#include "gymtest/airnets/aircnnnet.h"
#include "gymtest/airnets/airacbmnet.h"
#include "gymtest/airnets/airacnet.h"
#include "gymtest/airnets/airdueling.h"
#include "gymtest/airnets/airacbmsmallkernelnet.h"
#include "gymtest/lunarnets/cartacnet.h"
#include "gymtest/lunarnets/cartnet.h"
#include "gymtest/lunarnets/cartduelnet.h"
#include "gymtest/airnets/airachonet.h"
#include "gymtest/noisynets/noisycartfcnet.h"
#include "gymtest/noisynets/noisyaircnnnet.h"
#include "gymtest/train/rawpolicy.h"
#include "gymtest/train/softmaxpolicy.h"
#include "alg/dqnoption.h"

#include "probeenvs/ProbeEnvWrapper.h"

#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/simplelayout.h>
#include <log4cxx/logmanager.h>

#include <vector>
namespace {
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("noisydqntest"));
const torch::Device deviceType = torch::kCUDA;

void testProbe(const int epochNum) {
	const int batchSize = 1;
	const int inputNum = 4;
	const int envId = 5;
	const int outputNum = 2;

	const int envNum = batchSize;
	ProbeEnvWrapper env(inputNum, envId, envNum);

	NoisyCartFcNet model(inputNum, outputNum);
	model.to(deviceType);
	NoisyCartFcNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{envNum, 4};
    DqnOption option(inputShape, deviceType);
    //env
    option.envNum = envNum;
    option.envStep = 4;
    //target model
    option.targetUpdateStep = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 10000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 1;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //output
    option.multiLifes = false;
    //grad
    option.batchSize = 32;
    option.startStep = 1000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/noisy_testprobe/tfevents.pb";
    //test
    option.toTest = false;
    option.testGapEp = 500;
    //model
    option.saveModel = false;
    option.savePathPrefix = "???";
    option.loadModel = false;
    option.loadOptimizer = false;

    RawPolicy policy(option.exploreBegin, outputNum);

    NoisyDoubleDqn<NoisyCartFcNet, ProbeEnvWrapper, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);
    dqn.train(epochNum);
}


void testCart(const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 1; //8
	const int outputNum = 2;
	const int inputNum = 4;
	const int testClientNum = 4;

	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	LunarEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	NoisyCartFcNet model(inputNum, outputNum);
	model.to(deviceType);
	NoisyCartFcNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);

//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4};
    at::IntArrayRef testInputShape {testClientNum, 4};

    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = clientNum;
    option.envStep = 4;
    //target model
    option.targetUpdateStep = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 10000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 1;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //output
    option.multiLifes = false;
    //grad
    option.batchSize = 32;
    option.startStep = 1000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/noisy_testcart/tfevents.pb";
    //test
    option.toTest = true;
    option.testGapEp = 500;
    option.testBatch = testClientNum;
    option.testEp = testClientNum;
    //model
    option.saveModel = false;
    option.savePathPrefix = "???";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    NoisyDoubleDqn<NoisyCartFcNet, LunarEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);

    dqn.train(epochNum);
}


void testPong(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	const int testClientNum = 4;

	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");


	NoisyAirCnnNet model(outputNum);
	model.to(deviceType);
	NoisyAirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    at::IntArrayRef testInputShape{testClientNum, 4, 84, 84};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = clientNum;
    option.envStep = 4;
    //target model
    option.targetUpdateStep = 2000;
    option.tau = 1;
    //buffer
    option.rbCap = 300000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //output
    option.multiLifes = false;
    //grad
    option.batchSize = 128;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/noisy_testpong/tfevents.pb";
    //test
    option.toTest = true;
    option.testGapEp = 10000;
    option.testBatch = testClientNum;
    option.testEp = testClientNum;
    //model
    option.saveThreshold = -20;
    option.saveStep = 1;
    option.saveModel = false;
    option.savePathPrefix = "./noisydoubledqn_testpong1";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    NoisyDoubleDqn<NoisyAirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);

    dqn.train(epochNum);
}

//TODOED: envStep = 1
void testPong1(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	const int testClientNum = 4;

	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");


	NoisyAirCnnNet model(outputNum);
	model.to(deviceType);
	NoisyAirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    at::IntArrayRef testInputShape{testClientNum, 4, 84, 84};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = clientNum;
    option.envStep = 1;
    //target model
    option.targetUpdateStep = 2000;
    option.tau = 1;
    //buffer
    option.rbCap = 300000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //output
    option.multiLifes = false;
    //grad
    option.batchSize = 128;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/noisy_testpong1/tfevents.pb";
    //test
    option.toTest = true;
    option.testGapEp = 10000;
    option.testBatch = testClientNum;
    option.testEp = testClientNum;
    //model
    option.saveThreshold = -20;
    option.saveStep = 1;
    option.saveModel = false;
    option.savePathPrefix = "./noisydoubledqn_testpong1";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    NoisyDoubleDqn<NoisyAirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);

    dqn.train(epochNum);
}

//TODO: try smaller lr

void testPongLog(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	NoisyAirCnnNet model(outputNum);
	model.to(deviceType);
	NoisyAirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 10000;
    option.tau = 1;
    //buffer
    option.rbCap = 10240;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 256;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 128;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./noisydoubledqn_testponglog";
    //model
    option.saveThreshold = -19;
    option.saveStep = 1;
    option.saveModel = false;
    option.savePathPrefix = "./noisydoubledqn_testponglog";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    NoisyDoubleDqn<NoisyAirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}


void testPong2(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	NoisyAirCnnNet model(outputNum);
	model.to(deviceType);
	NoisyAirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 10240;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 256;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 128;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./noisydoubledqn_testpong2";
    //model
    option.saveThreshold = -20;
    option.saveStep = 1;
    option.saveModel = true;
    option.savePathPrefix = "./noisydoubledqn_testpong2";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    NoisyDoubleDqn<NoisyAirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}


void testPong3(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	NoisyAirCnnNet model(outputNum);
	model.to(deviceType);
	NoisyAirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 1024;
    option.tau = 1;
    //buffer
    option.rbCap = 100000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 256;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 32;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./noisydoubledqn_testpong3";
    //model
    option.saveThreshold = -20;
    option.saveStep = 1;
    option.saveModel = true;
    option.savePathPrefix = "./noisydoubledqn_testpong3";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    NoisyDoubleDqn<NoisyAirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}


void testtestPong(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	NoisyAirCnnNet model(outputNum);
	model.to(deviceType);
//	NoisyAirCnnNet targetModel(outputNum);
//	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 100;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 256;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 128;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./noisydoubledqn_testtestpong1";
    //model
    option.saveThreshold = -20;
    option.saveStep = 1;
    option.saveModel = false;
    option.savePathPrefix = "./noisydoubledqn_testtestpong1";
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/noisydoubledqn_testpong1_18.023436";



    RawPolicy policy(option.exploreBegin, outputNum);

    NoisyDoubleDqn<NoisyAirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, model, env, env, policy, optimizer, option);

    dqn.test(epochNum);
}



void testBreakout1(const int epochNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	NoisyAirCnnNet model(outputNum);
	model.to(deviceType);
	NoisyAirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 1024;
    option.tau = 1;
    //buffer
    option.rbCap = 100000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 256;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 128;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./noisydoubledqn_testbr1";
    //model
    option.saveThreshold = 1;
    option.saveStep = 1;
    option.saveModel = true;
    option.savePathPrefix = "./noisydoubledqn_testbr1";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    NoisyDoubleDqn<NoisyAirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}

//factor
void testBreakout2(const int epochNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	NoisyAirCnnNet model(outputNum);
	model.to(deviceType);
	NoisyAirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 100000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 256;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 32;
    option.startStep = 5000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./noisydoubledqn_testbr2";
    //model
    option.saveThreshold = 1;
    option.saveStep = 1;
    option.saveModel = true;
    option.savePathPrefix = "./noisydoubledqn_testbr2";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    NoisyDoubleDqn<NoisyAirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}

//factor
void testBreakout3(const int epochNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	NoisyAirCnnNet model(outputNum);
	model.to(deviceType);
	NoisyAirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 100000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 256;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 32;
    option.startStep = 5000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./noisydqn_testbr3";
    //model
    option.saveThreshold = 1;
    option.saveStep = 1;
    option.saveModel = true;
    option.savePathPrefix = "./noisydqn_testbr3";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    NoisyDqn<NoisyAirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}



void testBreakout4(const int epochNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	NoisyAirCnnNet model(outputNum);
	model.to(deviceType);
	NoisyAirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 1024;
    option.tau = 1;
    //buffer
    option.rbCap = 100000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 256;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 128;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./noisydoubledqn_testbr4";
    //model
    option.saveThreshold = 1;
    option.saveStep = 1;
    option.saveModel = true;
    option.savePathPrefix = "./noisydoubledqn_testbr4";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    NoisyDqn<NoisyAirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
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

//	testtestCart(atoi(argv[1]));
//	test103(atoi(argv[1]));
//	testPong(atoi(argv[1]));
//	testBreakout(atoi(argv[1]));

//	testCart(atoi(argv[1]));
//	testProbe(atoi(argv[1]));
	testPong1(atoi(argv[1]));

//	testDqnProbe(atoi(argv[1]));
//	testDqnCart(atoi(argv[1]));


//	testPongLog(atoi(argv[1]));

//	testtestCart(atoi(argv[1]));
//	testtestPong(atoi(argv[1]));
//	testBreakout4(atoi(argv[1]));


	LOG4CXX_INFO(logger, "End of test");
}
