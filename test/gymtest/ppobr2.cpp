/*
 * ppobr2.cpp
 *
 *  Created on: Aug 16, 2021
 *      Author: zf
 */




#include "alg/pposhared.hpp"
#include "alg/pposharedtest.hpp"
#include "alg/pporandom.hpp"
#include "alg/pponegreward.hpp"
#include "alg/pporecalc.hpp"

#include "gymtest/env/airenv.h"
#include "gymtest/env/lunarenv.h"
#include "gymtest/airnets/aircnnnet.h"
#include "gymtest/airnets/airacbmnet.h"
#include "gymtest/airnets/airacnet.h"
#include "gymtest/airnets/airacbmsmallkernelnet.h"
#include "gymtest/lunarnets/cartacnet.h"
#include "gymtest/airnets/airachonet.h"
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
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("pposharedbreakout"));
const torch::Device deviceType = torch::kCUDA;

void test0(const int updateNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 8; //8
	const int outputNum = 2;
	const int inputNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartACFcNet model(inputNum, outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 3; //from stat, seemed a reward feedback at first 25 steps
    const int roundNum = 2;

    at::IntArrayRef inputShape{clientNum, 4};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = false;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./ppobr2_test0";
    option.saveModel = false;
    option.savePathPrefix = "./ppobr2_test0";
    option.toTest = false;
    option.inputScale = 1;
    option.batchSize = maxStep;
    option.envNum = clientNum;
    option.epochNum = 8; //4
    option.trajStepNum = maxStep * roundNum; //200 //TODO:
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;

    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPORecalc<CartACFcNet, LunarEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}


void test00(const int updateNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 8; //8
	const int outputNum = 2;
	const int inputNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartACFcNet model(inputNum, outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 3; //from stat, seemed a reward feedback at first 25 steps
    const int roundNum = 2;

    at::IntArrayRef inputShape{clientNum, 4};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = false;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./ppobr2_test00";
    option.saveModel = false;
    option.savePathPrefix = "./ppobr2_test00";
    option.toTest = false;
    option.inputScale = 1;
    option.batchSize = maxStep;
    option.envNum = clientNum;
    option.epochNum = 8; //4
    option.trajStepNum = maxStep * roundNum; //200 //TODO:
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;

    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPORandom<CartACFcNet, LunarEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

void test1(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.0003).eps(1e-4));

    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 16; //from stat, seemed a reward feedback at first 25 steps
    const int roundNum = 8;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./ppobr2_test1";
    option.saveModel = true;
    option.savePathPrefix = "./ppobr2_test1";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 4;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.normReward = true;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.normReward = true;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test51";


    SoftmaxPolicy policy(outputNum);
    PPORecalc<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
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
//    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getDebug());
    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getInfo());
    log4cxx::LogManager::getLoggerRepository()->setConfigured(true);
}
}

int main(int argc, char** argv) {
	logConfigure(false);


//	test00(atoi(argv[1]));
	test1(atoi(argv[1]));


//	testOptOption();
//	testLoadOptLr();
}
