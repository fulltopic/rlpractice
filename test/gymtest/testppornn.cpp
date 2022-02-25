/*
 * testppornn.cpp
 *
 *  Created on: Feb 22, 2022
 *      Author: zf
 */



#include "alg/rnn/ppogrutruncslimgae.hpp"
#include "alg/utils/dqnoption.h"

#include "gymtest/env/airenv.h"
#include "gymtest/env/lunarenv.h"
#include "gymtest/rnnnets/lunarnets/cartacgrutruncnet.h"
#include "gymtest/rnnnets/lunarnets/cartacgruslim.h"
#include "gymtest/rnnnets/airnets/airacgrunet.h"
#include "gymtest/rnnnets/airnets/airacgruslimnet.h"

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
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("ppornnpong"));
const torch::Device deviceType = torch::kCUDA;


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

void testCart(const int updateNum) {
	const std::string envName = "CartPole-v0";
	const int envNum = 40;
	const int batchNum = 40; //8
	const int testClientNum = 4;
	const int outputNum = 2;
	const int inputNum = 4;
	const int hiddenLayerNum = 1;
	const int hiddenNum = 256;

	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, envNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10206";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	LunarEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartACGRUTruncFcSlimNet model(inputNum, hiddenNum, outputNum);
	model.to(deviceType);

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{4};
    at::IntArrayRef testInputShape {4};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = envNum;
    option.isAtari = false;
    option.donePerEp = 1;
    option.multiLifes = false;
    //grad
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.gamma = 0.99;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/ppornn_testcart_m8e8/tfevents.pb";
    //input
    option.inputScale = 1;
    option.batchSize = batchNum;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    //output
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    //ppo
    option.epochNum = 8; //4
    option.trajStepNum = batchNum * 1; //200 //TODO:
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    //test
    option.toTest = true;
    option.testGapEp = 1000;
    option.testBatch = testClientNum;
    option.testEp = testClientNum;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./???";
    option.loadModel = false;
    option.loadOptimizer = false;
    //rnn
    option.hiddenNums = {hiddenNum};
    option.hidenLayerNums = {1};
    option.maxStep = 8;
    option.gruCellNum = 1;

    SoftmaxPolicy policy(outputNum);
    PPOGRUTruncSlimGae<CartACGRUTruncFcSlimNet, LunarEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, testEnv, policy, optimizer, option);
    ppo.train(updateNum);
}
}

int main(int argc, char** argv) {
	logConfigure(false);

	testCart(atoi(argv[1]));
//	testPong(atoi(argv[1]));
//	testCartGae(atoi(argv[1]));
//	testPongGae(atoi(argv[1]));
//	testPongSlimGae20(atoi(argv[1]));
//	testBr(atoi(argv[1]));
//	testBrGae(atoi(argv[1]));
//	testCartSlim(atoi(argv[1]));
//	testPongSlim(atoi(argv[1]));


	return 0;
}
