/*
 * testdqn.cpp
 *
 *  Created on: Aug 17, 2021
 *      Author: zf
 */



#include "alg/dqnzip.hpp"

#include "gymtest/env/airenv.h"
#include "gymtest/env/lunarenv.h"
#include "gymtest/airnets/aircnnnet.h"
#include "gymtest/airnets/airacbmnet.h"
#include "gymtest/airnets/airacnet.h"
#include "gymtest/airnets/airacbmsmallkernelnet.h"
#include "gymtest/lunarnets/cartacnet.h"
#include "gymtest/lunarnets/cartnet.h"
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


void testCart(const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 1; //8
	const int outputNum = 2;
	const int inputNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	LunarEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartFcNet model(inputNum, outputNum);
	model.to(deviceType);
	CartFcNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);

//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(3e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 1000; //1000 worse than 2000
    option.tau = 1;
    //buffer
    option.rbCap = 81920;
    //explore
    option.exploreBegin = 0.5;
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
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/dqn_test0/tfevents.pb";
    //test
    option.toTest =  true;
    option.testGapEp = 10000;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./dqn_test0";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    DqnZip<CartFcNet, LunarEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);

    dqn.train(epochNum);
}

void testProbe(const int epochNum) {
	const int batchSize = 1;
	const int inputNum = 4;
	const int envId = 5;
	const int outputNum = 2;

	const int envNum = batchSize;
	ProbeEnvWrapper env(inputNum, envId, envNum);

	CartFcNet model(inputNum, outputNum);
	model.to(deviceType);
	CartFcNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(3e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{envNum, 4};
    DqnOption option(inputShape, deviceType);
    option.envNum = envNum;
    //target model
    option.targetUpdate = 10000;
    option.tau = 1;
    //buffer
    option.rbCap = 10240;
    //explore
    option.exploreBegin = 1;
    option.exploreEnd = 0.1;
    option.explorePart = 0.8;
    //input
    option.inputScale = 1;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 32;
    option.startStep = 500;
    option.maxGradNormClip = 1;
    //test
    option.toTest = true;
    option.testGapEp = 1000;
    option.testBatch = 1;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/dqn_testprobe/tfevents.pb";
    //model
    option.saveModel = false;
    option.savePathPrefix = "./dqn_test00";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    DqnZip<CartFcNet, ProbeEnvWrapper, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);

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
	std::string targetServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << targetServerAddr);
	AirEnv testEnv(targetServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(3e-4));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    at::IntArrayRef testInputShape{testClientNum, 4, 84, 84};
    DqnOption option(inputShape, testInputShape, deviceType);
    option.envNum = clientNum;
    //target model
    option.targetUpdate = 4000;
    option.tau = 1;
    //buffer
    option.rbCap = 300000;
    //explore
    option.envStep = 4;
    option.exploreBegin = 0.6;
    option.exploreEnd = 0.01;
    option.explorePart = 0.8;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //output
    option.multiLifes = false;
    //grad
    option.batchSize = 32;
    option.startStep = 100;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 400;
    option.tensorboardLogPath = "./logs/dqn_test1/tfevents.pb";
    //test
    option.toTest =  true;
    option.testGapEp = 10000;
    option.testBatch = testClientNum;
    option.testEp = testClientNum;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./dqn_test1";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    DqnZip<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);

    dqn.train(epochNum);
}


void testtestBreakout(const int epochNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);


    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 128;
    //explore
    option.exploreBegin = 0.5;
    option.exploreEnd = 0.01;
    option.explorePart = 0.8;
    //input
    option.inputScale = 256;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 32;
    option.startStep = 100;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
//    option.statCap = 128;
//    option.statPathPrefix = "./dqn_testtest121";
    option.tensorboardLogPath = "./logs/dqn_testtestbr/tfevents.pb";
    //model
    option.saveModel = false;
    option.savePathPrefix = "./dqn_testtest121";
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqn_test122";


    RawPolicy policy(option.exploreBegin, outputNum);

    DqnZip<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, model, env, env, policy, optimizer, option);
    dqn.test(epochNum);
}

void testBreakout(const int epochNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 1;
	const int testClientNum = 4;

	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string targetServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << targetServerAddr);
	AirEnv targetEnv(targetServerAddr, envName, testClientNum);
	targetEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(3e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    at::IntArrayRef testInputShape{testClientNum, 4, 84, 84};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = clientNum;
    option.multiLifes = true;
    option.livePerEpisode = 5;
    //target model
    option.targetUpdateStep = 5000;
    option.tau = 1;
    //buffer
    option.rbCap = 300000;
    //explore
    option.exploreBegin = 0.5;
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
    option.startStep = 10000;
    option.maxGradNormClip = 0.5;
    //test
    option.toTest = true;
    option.testGapEp = 10000;
    option.testBatch = testClientNum;
    option.testEp = testClientNum;
    //log
    option.logInterval = 400;
    option.tensorboardLogPath = "./logs/dqn_testbr/tfevents.pb";
    //model
    option.saveModel = false;
    option.savePathPrefix = "./dqn_test100";
    option.loadModel = false;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqn_test100";


    RawPolicy policy(option.exploreBegin, outputNum);

    DqnZip<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);
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
    log4cxx::LogManager::getLoggerRepository()->setConfigured(true);
}
}

int main(int argc, char** argv) {
	logConfigure(false);

	testPong(atoi(argv[1]));
//	testBreakout(atoi(argv[1]));
//	testProbe(atoi(argv[1]));



	LOG4CXX_INFO(logger, "End of test");
}
