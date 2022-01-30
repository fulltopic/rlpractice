/*
 * testppo.cpp
 *
 *  Created on: Jan 20, 2022
 *      Author: zf
 */



#include "alg/cnn/pposhared.hpp"
#include "alg/cnn/pporandom.hpp"
#include "alg/cnn/pporecalc.hpp"

#include "gymtest/env/airenv.h"
#include "gymtest/env/lunarenv.h"
#include "gymtest/cnnnets/airnets/aircnnnet.h"
#include "gymtest/cnnnets/airnets/airacbmnet.h"
#include "gymtest/cnnnets/airnets/airacnet.h"
#include "gymtest/cnnnets/airnets/airacbmsmallkernelnet.h"
#include "gymtest/cnnnets/lunarnets/cartacnet.h"
#include "gymtest/cnnnets/airnets/airachonet.h"
#include "gymtest/train/rawpolicy.h"
#include "gymtest/train/softmaxpolicy.h"
#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/simplelayout.h>
#include <log4cxx/logmanager.h>

#include <vector>
#include "alg/utils/dqnoption.h"

namespace {
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("pposharedpong"));
const torch::Device deviceType = torch::kCUDA;


void testSharedCart(const int updateNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 8; //8
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

	CartACFcNet model(inputNum, outputNum);
	model.to(deviceType);

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(3e-4));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4};
    at::IntArrayRef testInputShape {testClientNum, 4};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = clientNum;
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
    option.tensorboardLogPath = "./logs/pposhared_testcart/tfevents.pb";
    //input
    option.inputScale = 1;
    option.batchSize = 4;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    //output
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    //ppo
    option.epochNum = 4; //4
    option.trajStepNum = 200; //200 //TODO:
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

    SoftmaxPolicy policy(outputNum);
    PPOShared<CartACFcNet, LunarEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, testEnv, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

void testRandomCart(const int updateNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 8; //8
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

	CartACFcNet model(inputNum, outputNum);
	model.to(deviceType);

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(3e-4));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4};
    at::IntArrayRef testInputShape {testClientNum, 4};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = clientNum;
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
    option.tensorboardLogPath = "./logs/pporandom_testcart/tfevents.pb";
    //input
    option.inputScale = 1;
    option.batchSize = clientNum;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    //output
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    //ppo
    option.epochNum = 4; //4
    option.trajStepNum = 200; //200 //TODO:
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

    SoftmaxPolicy policy(outputNum);
    PPORandom<CartACFcNet, LunarEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, testEnv, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

void testRecalcCart(const int updateNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 8; //8
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

	CartACFcNet model(inputNum, outputNum);
	model.to(deviceType);

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(3e-4));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4};
    at::IntArrayRef testInputShape {testClientNum, 4};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = clientNum;
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
    option.tensorboardLogPath = "./logs/pporecalc_testcart/tfevents.pb";
    //input
    option.inputScale = 1;
    option.batchSize = clientNum;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    //output
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    //ppo
    option.epochNum = 4; //4
    option.trajStepNum = 200; //200 //TODO:
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

    SoftmaxPolicy policy(outputNum);
    PPORecalc<CartACFcNet, LunarEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, testEnv, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}


void testRandomPong(const int updateNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int clientNum = 50; //8
	const int testClientNum = 4;
	const int outputNum = 6;
//	const int inputNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(3e-4));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    at::IntArrayRef testInputShape{testClientNum, 4, 84, 84};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = clientNum;
    option.isAtari = true;
    option.donePerEp = 1;
    option.multiLifes = false;
    //grad
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
    option.gamma = 0.99;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/pporandom_testpong/tfevents.pb";
    //input
    option.inputScale = 255;
    option.batchSize = 10;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    //output
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    //ppo
    option.epochNum = 10; //4
    option.trajStepNum = option.batchSize * 10; //200 //TODO:
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

    SoftmaxPolicy policy(outputNum);
    PPORandom<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, testEnv, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}


void testRecalcPong(const int updateNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int clientNum = 50; //8
	const int testClientNum = 4;
	const int outputNum = 6;
//	const int inputNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(3e-4));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    at::IntArrayRef testInputShape{testClientNum, 4, 84, 84};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = clientNum;
    option.isAtari = true;
    option.donePerEp = 1;
    option.multiLifes = false;
    //grad
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
    option.gamma = 0.99;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/pporecalc_testpong/tfevents.pb";
    //input
    option.inputScale = 255;
    option.batchSize = 10;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    //output
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    //ppo
    option.epochNum = 10; //4
    option.trajStepNum = option.batchSize * 10; //200 //TODO:
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

    SoftmaxPolicy policy(outputNum);
    PPORecalc<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, testEnv, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}


void testRandomBr(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int clientNum = 50; //8
	const int testClientNum = 4;
	const int outputNum = 4;
//	const int inputNum = 4;
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

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(3e-4));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    at::IntArrayRef testInputShape{testClientNum, 4, 84, 84};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = clientNum;
    option.isAtari = true;
    option.donePerEp = 5;
    option.multiLifes = true;
    //grad
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
    option.gamma = 0.99;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/pporandom_testbr/tfevents.pb";
    //input
    option.inputScale = 255;
    option.batchSize = 20;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    //output
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    //ppo
    option.epochNum = 10; //4
    option.trajStepNum = option.batchSize * 10; //200 //TODO:
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

    SoftmaxPolicy policy(outputNum);
    PPORandom<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, testEnv, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}


void testRandomBr1(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int clientNum = 50; //8
	const int testClientNum = 4;
	const int outputNum = 4;
//	const int inputNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    at::IntArrayRef testInputShape{testClientNum, 4, 84, 84};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = clientNum;
    option.isAtari = true;
    option.donePerEp = 5;
    option.multiLifes = true;
    //grad
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.gamma = 0.99;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/pporandom_testbr1/tfevents.pb";
    //input
    option.inputScale = 255;
    option.batchSize = 16;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    //output
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    //ppo
    option.epochNum = 10; //4
    option.trajStepNum = option.batchSize * 8; //200 //TODO:
    option.ppoLambda = 0.9;
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

    SoftmaxPolicy policy(outputNum);
    PPORandom<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, testEnv, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
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
//    	log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getDebug());
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
	testRandomCart(atoi(argv[1]));
//	testSharedCart(atoi(argv[1]));
//	testRandomBr1(atoi(argv[1]));

	return 0;
}
