/*
 * dueltest.cpp
 *
 *  Created on: Apr 21, 2021
 *      Author: zf
 */



#include "alg/dqn.hpp"
#include "alg/nbrbdqn.hpp"
#include "alg/dqntarget.hpp"
#include "alg/dqntargetonline.hpp"
#include "alg/dqnsingle.hpp"
#include "alg/doubledqnsingle.hpp"

#include "gymtest/env/airenv.h"
#include "gymtest/airnets/airdueling.h"
#include "gymtest/train/rawpolicy.h"
#include "alg/dqnoption.h"


#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/logger.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/simplelayout.h>
#include <log4cxx/logmanager.h>

#include <vector>

namespace {
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("duelingtest"));
const torch::Device deviceType = torch::kCUDA;

void test0(const int batchSize, const int epochNum) {
	const std::string envName = "Breakout-v0";
	const int outputNum = 4;

	const int clientNum = 2;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnDuelNet model(outputNum);
	model.to(deviceType);
	AirCnnDuelNet targetModel(outputNum);
	targetModel.to(deviceType);
    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(0.00025).eps(1e-6).weight_decay(0.99).momentum(0));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 2048;
    option.statPath = "./stat_dqn_test15.txt";
    option.teststatPath = "test_dqn_test15";
    option.rbCap = 20480;
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test15";
    option.exploreBegin = 1;
    option.exploreDecay = 0.1;
    option.exploreEp = epochNum / 2;
    option.explorePhase =  10;
    option.loadModel = false;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0418/test7";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnDuelNet, AirEnv, RawPolicy, torch::optim::Adagrad> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum);
//    dqn.save();
}

void test1(const int batchSize, const int epochNum) {
	const std::string envName = "Breakout-v0";
	const int outputNum = 4;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, num);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnDuelNet model(outputNum);
	model.to(deviceType);
	AirCnnDuelNet targetModel(outputNum);
	targetModel.to(deviceType);
    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-4)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.gamma = 0.99;
    option.targetUpdate = 2048; //dqn paper recommended 10,000 frames, ddqn paper recommended 30,000 frames
    option.statPath = "./stat_duel_ddqn_test1.txt";
    option.teststatPath = "test_duel_ddqn_test1";
    option.rbCap = 20480;
    option.saveModel = true;
    option.savePathPrefix = "./duel_test1";
    option.exploreBegin = 1;
    option.exploreEnd = 0.1;
    option.exploreStep = epochNum * 10 / 2;
    option.explorePhase =  10;
    option.loadModel = false;

    RawPolicy policy(option.exploreBegin, outputNum);

    DDqnSingle<AirCnnDuelNet, AirEnv, RawPolicy, torch::optim::Adagrad> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum, 1);
//    dqn.save();

}

void test2(const int batchSize, const int epochNum) {
	const std::string envName = "Breakout-v0";
	const int outputNum = 4;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, num);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnDuelNet model(outputNum);
	model.to(deviceType);
	AirCnnDuelNet targetModel(outputNum);
	targetModel.to(deviceType);
    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-4)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.gamma = 0.99;
    option.targetUpdate = 2048; //dqn paper recommended 10,000 frames, ddqn paper recommended 30,000 frames
    option.statPath = "./stat_duel_ddqn_test2.txt";
    option.teststatPath = "test_duel_ddqn_test2";
    option.rbCap = 20480;
    option.saveModel = true;
    option.savePathPrefix = "./duel_test2";
    option.exploreBegin = 0.1;
    option.exploreEnd = 0.1;
    option.exploreStep = 1;
    option.explorePhase =  10;
    option.loadModel = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/duel_test1";


    RawPolicy policy(option.exploreBegin, outputNum);

    DDqnSingle<AirCnnDuelNet, AirEnv, RawPolicy, torch::optim::Adagrad> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum, 1);
//    dqn.save();

}

void test3(const int batchSize, const int epochNum) {
	const std::string envName = "Breakout-v0";
	const int outputNum = 4;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, num);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnDuelNet model(outputNum);
	model.to(deviceType);
	AirCnnDuelNet targetModel(outputNum);
	targetModel.to(deviceType);
    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-4)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.gamma = 0.99;
    option.targetUpdate = 2048; //dqn paper recommended 10,000 frames, ddqn paper recommended 30,000 frames
    option.statPath = "./stat_duel_ddqn_test3.txt";
    option.teststatPath = "test_duel_ddqn_test3";
    option.rbCap = 20480;
    option.saveModel = true;
    option.savePathPrefix = "./duel_test3";
    option.exploreBegin = 0.2;
    option.exploreEnd = 0.1;
    option.exploreStep = epochNum * 10 / 2;
    option.explorePhase =  10;
    option.loadModel = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/duel_test2";


    RawPolicy policy(option.exploreBegin, outputNum);

    DDqnSingle<AirCnnDuelNet, AirEnv, RawPolicy, torch::optim::Adagrad> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum, 1);
//    dqn.save();

}
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
    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getInfo());
    log4cxx::LogManager::getLoggerRepository()->setConfigured(true);
}
}

int main(int argc, char** argv) {
	logConfigure(false);
//	log4cxx::BasicConfigurator::configure();
//	log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getDebug());
//	log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getInfo());

//	test0(atoi(argv[1]), atoi(argv[2]));
//	test1(atoi(argv[1]), atoi(argv[2]));
//	test2(atoi(argv[1]), atoi(argv[2]));
	test3(atoi(argv[1]), atoi(argv[2]));
//	test4(atoi(argv[1]), atoi(argv[2]));
//	test5(atoi(argv[1]), atoi(argv[2]));
//	test6(atoi(argv[1]), atoi(argv[2]));
//	test7(atoi(argv[1]), atoi(argv[2]));
//	test8(atoi(argv[1]), atoi(argv[2]));
//	test9(atoi(argv[1]), atoi(argv[2]));
//	test10(atoi(argv[1]), atoi(argv[2]));
//	test11(atoi(argv[1]), atoi(argv[2]));
//	test12(atoi(argv[1]), atoi(argv[2]));
//	test13(atoi(argv[1]), atoi(argv[2]));
//	test14(atoi(argv[1]), atoi(argv[2]));
//	test15(atoi(argv[1]), atoi(argv[2]));

	LOG4CXX_INFO(logger, "End of test");
}
