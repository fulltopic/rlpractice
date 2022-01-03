/*
 * testpriodqn.cpp
 *
 *  Created on: Aug 26, 2021
 *      Author: zf
 */



#include "alg/priodqn.hpp"

#include "gymtest/env/airenv.h"
#include "gymtest/env/lunarenv.h"
#include "gymtest/airnets/aircnnnet.h"
#include "gymtest/airnets/airacbmnet.h"
#include "gymtest/airnets/airacnet.h"
#include "gymtest/airnets/airacbmsmallkernelnet.h"
#include "gymtest/lunarnets/cartacnet.h"
#include "gymtest/lunarnets/cartnet.h"
#include "gymtest/lunarnets/cartqnet.h"
#include "gymtest/airnets/airachonet.h"
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
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("dqntest"));
const torch::Device deviceType = torch::kCUDA;


void testProbe(const int epochNum) {
	const int batchSize = 1;
	const int inputNum = 4;
	const int envId = 5;
	const int outputNum = 2;

	const int envNum = batchSize;
	ProbeEnvWrapper env(inputNum, envId, envNum);
	ProbeEnvWrapper testEnv(inputNum, envId, envNum);

	CartFcQNet model(inputNum, outputNum);
	model.to(deviceType);
	CartFcQNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{envNum, 4};
    DqnOption option(inputShape, deviceType);
    //env
    option.envNum = envNum;
    option.envStep = 4;
    option.multiLifes = false;
    //target model
    option.targetUpdateStep = 100;
    option.tau = 1;
    //prio
    option.rbCap = 1024 * 8; //Cap to be 2^x
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.4;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1e-6;
    option.pbBetaPart = 1;
    //explore
    option.exploreBegin = 1;
    option.exploreEnd = 0.01;
    option.explorePart = 0.8;
    //input
    option.inputScale = 1;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 16;
    option.startStep = 100;
    option.maxGradNormClip = 1;
    //test
    option.toTest = true;
    option.testGapEp = 1000;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/prio_testprobe/tfevents.pb";
    //model
    option.saveModel = false;
    option.savePathPrefix = "???";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<CartFcQNet, ProbeEnvWrapper, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(epochNum);
}


void testCart(const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 1; //8
	const int outputNum = 2;
	const int inputNum = 4;
	const int testClientNum = 4;

	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	LunarEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartFcQNet model(inputNum, outputNum);
	model.to(deviceType);
	CartFcQNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(3e-4));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4};
    at::IntArrayRef testInputShape{testClientNum, 4};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = clientNum;
    option.envStep = 4;
    option.multiLifes = false;
    //target model
    option.targetUpdateStep = 200;
    option.tau = 1;
    //prio
    option.rbCap = 8192 * 8; //Cap to be 2^x
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.4;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1e-6;
    option.pbBetaPart = 0.9;
    //explore
    option.exploreBegin = 1;
    option.exploreEnd = 0.01;
    option.explorePart = 0.8;
    //input
    option.inputScale = 1;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 32;
    option.startStep = 1000;
    option.maxGradNormClip = 1;
    //test
    option.toTest = true;
    option.testGapEp = 1000;
    option.testEp = testClientNum;
    option.testBatch = testClientNum;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/prio_testcart/tfevents.pb";
    //model
    option.saveModel = false;
    option.savePathPrefix = "???";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<CartFcQNet, LunarEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);

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

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(3e-4));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    at::IntArrayRef testInputShape{testClientNum, 4, 84, 84};
    DqnOption option(inputShape, testInputShape, deviceType);

    //env
    option.envNum = clientNum;
    option.envStep = 8;
    option.multiLifes = false;
    //target model
    option.targetUpdateStep = 1000;
    option.tau = 1;
    //prio
    option.rbCap = 262144; //Cap to be 2^x
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.4;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1e-6;
    option.pbBetaPart = 0.9;
    //explore
    option.exploreBegin = 0.6;
    option.exploreEnd = 0.01;
    option.explorePart = 0.8;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 8192;
    option.maxGradNormClip = 1;
    //test
    option.toTest = true;
    option.testGapEp = 5000;
    option.testEp = testClientNum;
    option.testBatch = testClientNum;
    //log
    option.logInterval = 1000;
    option.tensorboardLogPath = "./logs/prio_testpong/tfevents.pb";
    //model
    option.saveThreshold = -20;
    option.saveStep = 1;
    option.saveModel = false;
    option.savePathPrefix = "???";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);

    dqn.train(epochNum);
}


void testPongDouble(const int epochNum) {
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

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(3e-4));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    at::IntArrayRef testInputShape{testClientNum, 4, 84, 84};
    DqnOption option(inputShape, testInputShape, deviceType);

    //env
    option.envNum = clientNum;
    option.envStep = 8;
    option.multiLifes = false;
    //target model
    option.targetUpdateStep = 2000;
    option.tau = 1;
    //prio
    option.rbCap = 262144; //Cap to be 2^x, 262144
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.4;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1e-5;
    option.pbBetaPart = 0.9;
    //explore
    option.exploreBegin = 0.8;
    option.exploreEnd = 0.01;
    option.explorePart = 0.8;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64; //32
    option.startStep = 1000; //1000
    option.maxGradNormClip = 1;
    //test
    option.toTest = true;
    option.testGapEp = 5000;
    option.testEp = testClientNum;
    option.testBatch = testClientNum;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/prio_testpongdouble/tfevents.pb";
    //model
    option.saveThreshold = -20;
    option.saveStep = 1;
    option.saveModel = false;
    option.savePathPrefix = "???";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);

    dqn.train(epochNum);
}

//TODO: envStep = 1; lr = 1e-4; step = 1000000
void testPongDouble1(const int epochNum) {
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

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(3e-4));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    at::IntArrayRef testInputShape{testClientNum, 4, 84, 84};
    DqnOption option(inputShape, testInputShape, deviceType);

    //env
    option.envNum = clientNum;
    option.envStep = 8;
    option.multiLifes = false;
    //target model
    option.targetUpdateStep = 2000;
    option.tau = 1;
    //prio
    option.rbCap = 262144; //Cap to be 2^x, 262144
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.4;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1e-5;
    option.pbBetaPart = 0.3;
    //explore
    option.exploreBegin = 1;
    option.exploreEnd = 0.01;
    option.explorePart = 0.3;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64; //32
    option.startStep = 1000; //1000
    option.maxGradNormClip = 1;
    //test
    option.toTest = true;
    option.testGapEp = 5000;
    option.testEp = testClientNum;
    option.testBatch = testClientNum;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/prio_testpongdouble1/tfevents.pb";
    //model
    option.saveThreshold = -20;
    option.saveStep = 1;
    option.saveModel = false;
    option.savePathPrefix = "???";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);

    dqn.train(epochNum);
}


void testPongDouble2(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
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

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    at::IntArrayRef testInputShape{testClientNum, 4, 84, 84};
    DqnOption option(inputShape, testInputShape, deviceType);

    //env
    option.envNum = clientNum;
    option.envStep = 1;
    option.multiLifes = false;
    //target model
    option.targetUpdateStep = 2000;
    option.tau = 1;
    //prio
    option.rbCap = 131072; //Cap to be 2^x, 262144
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.4;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1e-5;
    option.pbBetaPart = 0.3;
    //explore
    option.exploreBegin = 1;
    option.exploreEnd = 0.01;
    option.explorePart = 0.3;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 32; //32
    option.startStep = 10000; //1000
    option.maxGradNormClip = 1;
    //test
    option.toTest = true;
    option.testGapEp = 5000;
    option.testEp = testClientNum;
    option.testBatch = testClientNum;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/prio_testpongdouble2/tfevents.pb";
    //model
    option.saveThreshold = -20;
    option.saveStep = 1;
    option.saveModel = false;
    option.savePathPrefix = "???";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);

    dqn.train(epochNum);
}


void testBeamRider(const int epochNum) {
	const std::string envName = "BeamRiderNoFrameskip-v4";
	const int outputNum = 9;
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

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(3e-4));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    at::IntArrayRef testInputShape{testClientNum, 4, 84, 84};
    DqnOption option(inputShape, testInputShape, deviceType);

    //env
    option.envNum = clientNum;
    option.envStep = 8;
    option.multiLifes = false;
    //target model
    option.targetUpdateStep = 2000;
    option.tau = 1;
    //prio
    option.rbCap = 262144; //Cap to be 2^x, 262144
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.4;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1e-5;
    option.pbBetaPart = 0.9;
    //explore
    option.exploreBegin = 1;
    option.exploreEnd = 0.05;
    option.explorePart = 0.8;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64; //32
    option.startStep = 1000; //1000
    option.maxGradNormClip = 1;
    //test
    option.toTest = true;
    option.testGapEp = 5000;
    option.testEp = testClientNum;
    option.testBatch = testClientNum;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/prio_testbeam/tfevents.pb";
    //model
    option.saveThreshold = -20;
    option.saveStep = 1;
    option.saveModel = false;
    option.savePathPrefix = "???";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);

    dqn.train(epochNum);
}

//TODO: increase beta
//TODO: sometimes, beta > 1 makes better performance, but beta is not encouraged to > 1
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
//	testPong6(atoi(argv[1]));
//	testBreakout(atoi(argv[1]));

//	testCart(atoi(argv[1]));
//	testProbe(atoi(argv[1]));
	testPongDouble2(atoi(argv[1]));
//	testBeamRider(atoi(argv[1]));
//	testtestPong(atoi(argv[1]), argv[2]);
//	testCartLog(atoi(argv[1]));

	LOG4CXX_INFO(logger, "End of test");
}
