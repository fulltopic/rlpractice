/*
 * testa2crnn.cpp
 *
 *  Created on: Jan 27, 2022
 *      Author: zf
 */



#include "alg/rnn/a2cnrnn.hpp"
#include "alg/utils/dqnoption.h"

#include "gymtest/env/airenv.h"
#include "gymtest/env/lunarenv.h"
#include "gymtest/rnnnets/lunarnets/cartacrnnnet.h"

#include "gymtest/train/rawpolicy.h"
#include "gymtest/train/softmaxpolicy.h"

#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/simplelayout.h>
#include <log4cxx/logmanager.h>

#include <vector>

namespace {
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("a2cnbatchpong"));
const torch::Device deviceType = torch::kCUDA;

void testCart(const int epochNum) {
	const int batchSize = 40;
	const std::string envName = "CartPole-v0";
	const int outputNum = 2;
	const int inputNum = 4;
	const int testClientNum = 4;
	const int maxStep = 8;
	const int hiddenLayerNum = 1;
	const int hiddenNum = 512;

	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, batchSize);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	LunarEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartACRNNFcNet model(inputNum, hiddenNum, outputNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
    LOG4CXX_INFO(logger, "Model ready");


    at::IntArrayRef inputShape{batchSize, 4};
    at::IntArrayRef testInputShape {1, testClientNum, 4};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = batchSize;
    option.isAtari = false;
    option.envStep = maxStep;
    option.donePerEp = 1;
    option.multiLifes = false;
    //grad
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.gamma = 0.99;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/a2crnn_testcart/tfevents.pb";
    //input
    option.inputScale = 1;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    //test
    option.toTest = true;
    option.testGapEp = 400;
    option.testBatch = testClientNum;
    option.testEp = testClientNum;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./a2cngae_test0";
    //rnn
    option.hiddenNums = {hiddenNum};
    option.hidenLayerNums = {1};


    SoftmaxPolicy policy(outputNum);
    A2CNRNN<CartACRNNFcNet, LunarEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, testEnv, policy, optimizer, maxStep, option);
    a2c.train(epochNum);
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
    //	log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getDebug());
    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getInfo());
    log4cxx::LogManager::getLoggerRepository()->setConfigured(true);
}
}

int main(int argc, char** argv) {
	logConfigure(false);

//	testCart(atoi(argv[1]));
//	testPong1(atoi(argv[1]));
//	testCartGae(atoi(argv[1]));
//	testPongGae(atoi(argv[1]));
//	testBr(atoi(argv[1]));
	testCart(atoi(argv[1]));

	return 0;
}
