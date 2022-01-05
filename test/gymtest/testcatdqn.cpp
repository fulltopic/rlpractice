/*
 * testcatdqn.cpp
 *
 *  Created on: Oct 23, 2021
 *      Author: zf
 */


#include "alg/catdqn.hpp"

#include "gymtest/env/airenv.h"
#include "gymtest/env/lunarenv.h"
#include "gymtest/lunarnets/cartacnet.h"
#include "gymtest/lunarnets/cartqnet.h"
#include "gymtest/airnets/aircnnnet.h"
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
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("catdqn"));
const torch::Device deviceType = torch::kCUDA;

void testProbe(const int epochNum) {
	const int batchSize = 1;
	const int inputNum = 4;
	const int envId = 5;
	const int outputNum = 2;
	const int atomNum = 5;
	const int envNum = batchSize;

	ProbeEnvWrapper env(inputNum, envId, envNum);
	ProbeEnvWrapper testEnv(inputNum, envId, envNum);

	CartFcQNet model(inputNum, outputNum * atomNum);
	model.to(deviceType);
	CartFcQNet targetModel(inputNum, outputNum * atomNum);
	targetModel.to(deviceType);

    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{envNum, 4};
    DqnOption option(inputShape, deviceType);
    //env
    option.envNum = envNum;
    option.envStep = 1;
    //target model
    option.targetUpdate = 100;
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
    //catdqn
    option.vMin = 0;
    option.vMax = 4;
    option.outputNum = outputNum;
    option.atomNum = atomNum;
    //log
    option.tensorboardLogPath = "./logs/catdqn_testprobe/tfevents.pb";
    option.logInterval = 100;
    //test
    option.toTest = true;
    option.testEp = envNum;
    option.testGapEp = 100;
    option.testBatch = envNum;
	option.livePerEpisode = 1;
    //model
    option.saveModel = false;
    option.loadModel = false;
    option.loadOptimizer = false;
    option.loadPathPrefix = "???";


    RawPolicy policy(option.exploreBegin, outputNum);

    CategoricalDqn<CartFcQNet, ProbeEnvWrapper, RawPolicy, torch::optim::RMSprop> dqn(
    		model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(epochNum);
}

void testCart(const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 1; //8
	const int testClientNum = 4;
	const int outputNum = 2;
	const int inputNum = 4;
	const int atomNum = 51;

	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	LunarEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");


	CartFcQNet model(inputNum, outputNum * atomNum);
	model.to(deviceType);
	CartFcQNet targetModel(inputNum, outputNum * atomNum);
	targetModel.to(deviceType);

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
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
    //update
    option.envStep = 1;
    //grad
    option.batchSize = 4;
    option.startStep = 1000;
    option.maxGradNormClip = 1;
    //test
    option.toTest = true;
    option.testEp = testClientNum;
    option.testGapEp = 1000;
    option.testBatch = testClientNum;
	option.livePerEpisode = 1;
    //log
    option.tensorboardLogPath = "./logs/catdqn_testcart/tfevents.pb";
    option.logInterval = 100;
    //model
    option.saveThreshold = 5;
    option.saveStep = 1;
    option.saveModel = false;
    option.savePathPrefix = "./catdqn_testcart";
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqnzip_test127";
    //catdqn
    option.vMin = -10;
    option.vMax = 10;
    option.outputNum = outputNum;
    option.atomNum = atomNum;

    RawPolicy policy(option.exploreBegin, outputNum);

    CategoricalDqn<CartFcQNet, LunarEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(epochNum);
}

void testPong0(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int clientNum = 1; //8
	const int testClientNum = 6;
	const int outputNum = 6;
	const int atomNum = 51;

	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirCnnNet model(outputNum * atomNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum * atomNum);
	targetModel.to(deviceType);

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
    LOG4CXX_INFO(logger, "Model ready");


    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    at::IntArrayRef testInputShape{testClientNum, 4, 84, 84};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 2000;
    option.tau = 1;
    //buffer
    option.rbCap = 300000;
    //explore: step = 3,000,000
    option.exploreBegin = 1;
    option.exploreEnd = 0.01;
    option.explorePart = 0.3;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //update
    option.envStep = 8;
    option.epochPerUpdate = 1;
    //grad
    option.batchSize = 128;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //test
    option.toTest = true;
    option.testEp = testClientNum;
    option.testGapEp = 10000;
    option.testBatch = testClientNum;
    //log
    option.tensorboardLogPath = "./logs/catdqn_testpong0/tfevents.pb";
    option.logInterval = 100;
    //model
    option.saveModel = false;
    option.loadModel = false;
    //catdqn
    option.vMin = -10;
    option.vMax = 10;
    option.outputNum = outputNum;
    option.atomNum = atomNum;

    RawPolicy policy(option.exploreBegin, outputNum);

    CategoricalDqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(epochNum);
}

void testPong01(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int clientNum = 1; //8
	const int testClientNum = 6;
	const int outputNum = 6;
	const int atomNum = 51;
	std::string serverAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");
	std::string testServerAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirCnnNet model(outputNum * atomNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum * atomNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 10000;
    option.tau = 1;
    //buffer
    option.rbCap = 300000;
    //explore: step = 1,000,000
    option.exploreBegin = 0.05;
    option.exploreEnd = 0.01;
    option.explorePart = 0.6;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //update
    option.envStep = 16;
    option.epochPerUpdate = 4;
    //grad
    option.batchSize = 32;
    option.startStep = 20000;
    option.maxGradNormClip = 1;
    //test
    option.toTest = true;
    option.testEp = testClientNum;
    option.testGapEp = 10000;
    option.testBatch = testClientNum;
	option.livePerEpisode = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./catdqn_testpong01";
    option.tensorboardLogPath = "./logs/catdqn_testpong01/tfevents.pb";
    option.logInterval = 1000;
    //model
    option.saveThreshold = 15;
    option.saveStep = 1;
    option.saveModel = true;
    option.savePathPrefix = "./catdqn_testpong01";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/catdqn_testpong0";
    //catdqn
    option.vMin = -10;
    option.vMax = 10;
    option.outputNum = outputNum;
    option.atomNum = atomNum;

    RawPolicy policy(option.exploreBegin, outputNum);

    CategoricalDqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(epochNum);
}

void testtestPong(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int clientNum = 1; //8
	const int testClientNum = 6;
	const int outputNum = 6;
	const int atomNum = 51;
	std::string serverAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");
	std::string testServerAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirCnnNet model(outputNum * atomNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum * atomNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 4000;
    option.tau = 1;
    //buffer
    option.rbCap = 300000;
    //explore: step = 3,000,000
    option.exploreBegin = 1;
    option.exploreEnd = 0.05;
    option.explorePart = 0.6;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //update
    option.envStep = 8;
    option.epochPerUpdate = 1;
    //grad
    option.batchSize = 32;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //test
    option.toTest = true;
    option.testEp = 24;
    option.testGapEp = 10000;
    option.testBatch = testClientNum;
	option.livePerEpisode = 1;
    //log
//    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./catdqn_testtestpong";
    option.tensorboardLogPath = "./logs/catdqn_testtestpong/tfevents.pb";
    option.logInterval = 100;
    //model
    option.saveThreshold = 5;
    option.saveStep = 1;
    option.saveModel = false;
    option.savePathPrefix = "./catdqn_testpong0";
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/catdqn_testpong0";
    //catdqn
    option.vMin = -10;
    option.vMax = 10;
    option.outputNum = outputNum;
    option.atomNum = atomNum;

    RawPolicy policy(option.exploreBegin, outputNum);

    CategoricalDqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.test(epochNum, true, true);
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

//	test128(atoi(argv[1]));
//	test124(atoi(argv[1]));
	testPong0(atoi(argv[1]));
//	testBreakout(atoi(argv[1]));
//	testProbe(atoi(argv[1]));
//	testCart(atoi(argv[1]));

//	testtestPong(atoi(argv[1]));

	LOG4CXX_INFO(logger, "End of test");
}

