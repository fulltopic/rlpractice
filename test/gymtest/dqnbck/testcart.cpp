/*
 * testcart.cpp
 *
 *  Created on: Apr 24, 2021
 *      Author: zf
 */


#include "alg/dqnsingle.hpp"
#include "alg/dqn.hpp"
#include "alg/dqntarget.hpp"
#include "alg/dqnoption.h"
#include "alg/doubledqnsingle.hpp"


#include "gymtest/env/airenv.h"
#include "gymtest/env/lunarenv.h"
#include "gymtest/airnets/aircnnnet.h"
#include "gymtest/airnets/aircnnbmnet.h"
#include "gymtest/lunarnets/cartnet.h"
#include "gymtest/lunarnets/cartduelnet.h"
#include "gymtest/train/rawpolicy.h"
#include "gymtest/utils/nobatchrb.h"



#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/simplelayout.h>
#include <log4cxx/logmanager.h>

#include <vector>


namespace {
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("carttest"));
const torch::Device deviceType = torch::kCUDA;

void test0(const int batchSize, const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int outputNum = 2;
	const int inputNum = 4;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	LunarEnv testEnv(testServerAddr, envName, num);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartFcNet model(inputNum, outputNum);
	model.to(deviceType);
	CartFcNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4};
    DqnOption option(inputShape, deviceType);
    option.gamma = 0.99;
    option.targetUpdate = 1024;
    option.statPath = "./stat_cart_test0.txt";
    option.teststatPath = "./test_cart_test0.txt";
    option.rbCap = 10240;
    option.saveModel = false;
    option.savePathPrefix = "./cart_test0";
    option.exploreBegin = 1;
    option.exploreEnd = 0.1;
//    option.exploreDecay = 0.1;
    option.exploreStep = epochNum * 30 / 2;
//    option.explorePhase =  10;
    option.loadModel = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0418/test7";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<CartFcNet, LunarEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum);
//    dqn.save();
}

void test1(const int batchSize, const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int outputNum = 2;
	const int inputNum = 4;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	LunarEnv testEnv(testServerAddr, envName, num);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartFcNet model(inputNum, outputNum);
	model.to(deviceType);
	CartFcNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4};
    DqnOption option(inputShape, deviceType);
    option.gamma = 0.99;
    option.targetUpdate = 1024;
    option.statPath = "./stat_cart_test1.txt";
    option.teststatPath = "./test_cart_test1.txt";
    option.rbCap = 10240;
    option.saveModel = false;
    option.savePathPrefix = "./cart_test1";
    option.exploreBegin = 1;
    option.exploreEnd = 0.1;
//    option.exploreDecay = 0.1;
    option.exploreStep = epochNum * 30 / 2;
//    option.explorePhase =  10;
    option.loadModel = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0418/test7";

    RawPolicy policy(option.exploreBegin, outputNum);

    DDqnSingle<CartFcNet, LunarEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum);
//    dqn.save();
}

void test2(const int batchSize, const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int outputNum = 2;
	const int inputNum = 4;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	LunarEnv testEnv(testServerAddr, envName, num);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartDuelFcNet model(inputNum, outputNum);
	model.to(deviceType);
	CartDuelFcNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4};
    DqnOption option(inputShape, deviceType);
    option.gamma = 0.99;
    option.targetUpdate = 1024;
    option.statPath = "./stat_cart_test2.txt";
    option.teststatPath = "./test_cart_test2.txt";
    option.rbCap = 10240;
    option.saveModel = false;
    option.savePathPrefix = "./cart_test2";
    option.exploreBegin = 1;
    option.exploreEnd = 0.1;
//    option.exploreDecay = 0.1;
    option.exploreStep = epochNum * 30 / 2;
//    option.explorePhase =  10;
    option.loadModel = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0418/test7";

    RawPolicy policy(option.exploreBegin, outputNum);

    DDqnSingle<CartDuelFcNet, LunarEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum);
//    dqn.save();
}

void test3(const int batchSize, const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int outputNum = 2;
	const int inputNum = 4;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	LunarEnv testEnv(testServerAddr, envName, num);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartFcNet model(inputNum, outputNum);
	model.to(deviceType);
	CartFcNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4};
    DqnOption option(inputShape, deviceType);
    option.gamma = 0.99;
    option.targetUpdate = 1024;
    option.statPath = "./stat_cart_test3.txt";
    option.teststatPath = "./test_cart_test3.txt";
    option.rbCap = 20480;
    option.saveModel = false;
    option.savePathPrefix = "./cart_test3";
    option.exploreBegin = 1;
    option.exploreEnd = 0.1;
//    option.exploreDecay = 0.1;
    option.exploreStep = epochNum * 30 / 2;
//    option.explorePhase =  10;
    option.loadModel = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0418/test7";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<CartFcNet, LunarEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum);
//    dqn.save();
}

void test4(const int batchSize, const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int outputNum = 2;
	const int inputNum = 4;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	LunarEnv testEnv(testServerAddr, envName, num);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartFcNet model(inputNum, outputNum);
	model.to(deviceType);
	CartFcNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4};
    DqnOption option(inputShape, deviceType);
    option.gamma = 0.99;
    option.targetUpdate = 1024;
    option.statPath = "./stat_cart_test4.txt";
    option.teststatPath = "./test_cart_test4.txt";
    option.rbCap = 40960;
    option.saveModel = false;
    option.savePathPrefix = "./cart_test4";
    option.exploreBegin = 1;
    option.exploreEnd = 0.1;
//    option.exploreDecay = 0.1;
    option.exploreStep = epochNum * 30 / 2;
//    option.explorePhase =  10;
    option.loadModel = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0418/test7";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<CartFcNet, LunarEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum);
//    dqn.save();
}

void test5(const int batchSize, const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int outputNum = 2;
	const int inputNum = 4;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	LunarEnv testEnv(testServerAddr, envName, num);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartFcNet model(inputNum, outputNum);
	model.to(deviceType);
	CartFcNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4};
    DqnOption option(inputShape, deviceType);
    option.gamma = 0.99;
    option.targetUpdate = 1024;
    option.statPath = "./stat_cart_test5.txt";
    option.teststatPath = "./test_cart_test5.txt";
    option.rbCap = 40960;
    option.saveModel = false;
    option.savePathPrefix = "./cart_test5";
    option.exploreBegin = 1;
    option.exploreEnd = 0.1;
//    option.exploreDecay = 0.1;
    option.exploreStep = epochNum * 30 / 2;
//    option.explorePhase =  10;
    option.loadModel = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0418/test7";

    RawPolicy policy(option.exploreBegin, outputNum);

    DDqnSingle<CartFcNet, LunarEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum);
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
    //	log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getDebug());
    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getInfo());
    log4cxx::LogManager::getLoggerRepository()->setConfigured(true);
}
}

int main(int argc, char** argv) {
	logConfigure(false);

//	test0(atoi(argv[1]), atoi(argv[2]));
//	test1(atoi(argv[1]), atoi(argv[2]));
//	test2(atoi(argv[1]), atoi(argv[2]));
//	test3(atoi(argv[1]), atoi(argv[2]));
//	test4(atoi(argv[1]), atoi(argv[2]));
	test5(atoi(argv[1]), atoi(argv[2]));
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
//	test16(atoi(argv[1]), atoi(argv[2]));
//	test17(atoi(argv[1]), atoi(argv[2]));
//	test18(atoi(argv[1]), atoi(argv[2]));
//	test19(atoi(argv[1]), atoi(argv[2]));
//	test23(atoi(argv[1]), atoi(argv[2]));
//	testCal(atoi(argv[1]), atoi(argv[2]));

	LOG4CXX_INFO(logger, "End of test");
}
