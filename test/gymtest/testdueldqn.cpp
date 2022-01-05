/*
 * testdueldqn.cpp
 *
 *  Created on: Aug 24, 2021
 *      Author: zf
 */



#include "alg/dqndouble.hpp"
#include "alg/dqnzip.hpp"

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
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("dueldqntest"));
const torch::Device deviceType = torch::kCUDA;

void testProbe(const int epochNum) {
	const int batchSize = 1;
	const int inputNum = 4;
	const int envId = 5;
	const int outputNum = 2;

	const int envNum = batchSize;
	ProbeEnvWrapper env(inputNum, envId, envNum);
	ProbeEnvWrapper testEnv(inputNum, envId, envNum);

	CartDuelFcNet model(inputNum, outputNum);
	model.to(deviceType);
	CartDuelFcNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{envNum, 4};
    DqnOption option(inputShape, deviceType);
    option.envNum = envNum;
    option.envStep = 1;
    //target model
    option.targetUpdateStep = 100;
    option.tau = 1;
    //buffer
    option.rbCap = 1024;
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
    option.batchSize = 4;
    option.startStep = 100;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/duel_testprobe/tfevents.pb";
    //test
    option.toTest = true;
    option.testGapEp = 100;
    option.testBatch = envNum;
    option.testEp = envNum;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./doubledqn_testprobe";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    DoubleDqn<CartDuelFcNet, ProbeEnvWrapper, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(epochNum);
}


void testCart(const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 1; //8
	const int testClientNum = 4;
	const int outputNum = 2;
	const int inputNum = 4;

	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	LunarEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartDuelFcNet model(inputNum, outputNum);
	model.to(deviceType);
	CartDuelFcNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(2.5e-4));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4};
    at::IntArrayRef testInputShape {testClientNum, 4};

    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = clientNum;
    option.envStep = 1;
    //target model
    option.targetUpdateStep = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 10000;
    //explore
    option.exploreBegin = 1;
    option.exploreEnd = 0.01;
    option.explorePart = 0.6;
    //input
    option.inputScale = 1;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 4;
    option.startStep = 1000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/duel_testcart/tfevents.pb";
    //test
    option.toTest = true;
    option.testGapEp = 1000;
    option.testEp = testClientNum;
    option.testBatch = testClientNum;
    //model
    option.saveModel = false;

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnZip<CartDuelFcNet, LunarEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);

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

	AirCnnDuelNet model(outputNum);
	model.to(deviceType);
	AirCnnDuelNet targetModel(outputNum);
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
    option.batchSize = 128;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/duel_testpong/tfevents.pb";
    //test
    option.toTest = true;
    option.testGapEp = 10000;
    option.testBatch = testClientNum;
    option.testEp = testClientNum;
    //model
//    option.saveThreshold = 10;
//    option.saveStep = 1;
    option.saveModel = false;
    option.savePathPrefix = "???";
//    option.loadModel = false;
//    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    //TODO: Try Dqnzip, dqn is no worse than double in pong game
    DoubleDqn<AirCnnDuelNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);

    dqn.train(epochNum);
}


void testPongDqn(const int epochNum) {
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

	AirCnnDuelNet model(outputNum);
	model.to(deviceType);
	AirCnnDuelNet targetModel(outputNum);
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
    option.batchSize = 128;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/duel_testpongdqn/tfevents.pb";
    //test
    option.toTest = true;
    option.testGapEp = 10000;
    option.testBatch = testClientNum;
    option.testEp = testClientNum;
    //model
//    option.saveThreshold = 10;
//    option.saveStep = 1;
    option.saveModel = false;
    option.savePathPrefix = "???";
//    option.loadModel = false;
//    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    //TODO: Try Dqnzip, dqn is no worse than double in pong game
    DqnZip<AirCnnDuelNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);

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
	testPongDqn(atoi(argv[1]));
//	testBreakout(atoi(argv[1]));

//	testCart(atoi(argv[1]));
//	testProbe(atoi(argv[1]));
//	testPong7(atoi(argv[1]));

//	testtestPong(atoi(argv[1]));
//	testtestCart(atoi(argv[1]));

	LOG4CXX_INFO(logger, "End of test");
}

