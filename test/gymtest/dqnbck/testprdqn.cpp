/*
 * testprdqn.cpp
 *
 *  Created on: May 24, 2021
 *      Author: zf
 */


#include "alg/priorbdqn.hpp"
#include "alg/nbrbdqn.hpp"
#include "alg/dqntarget.hpp"
#include "alg/dqntargetonline.hpp"
#include "alg/dqnsingle.hpp"

#include "gymtest/env/airenv.h"
#include "gymtest/env/lunarenv.h"
#include "gymtest/airnets/aircnnnet.h"
#include "gymtest/airnets/aircnnbmnet.h"
#include "gymtest/airnets/airsmallkernelnet.h"
#include "gymtest/lunarnets/cartnet.h"
#include "gymtest/train/rawpolicy.h"
#include "alg/dqnoption.h"

#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/simplelayout.h>
#include <log4cxx/logmanager.h>

#include <vector>

namespace {
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("prdqntest"));
const torch::Device deviceType = torch::kCUDA;

void testCartpole(const int batchSize, const int totalStep) {
	const std::string envName = "CartPole-v0";
//	const std::string envName = "CartPoleNoFrameskip-v4";
	const int outputNum = 2;
	const int inputNum = 4;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, num);
	env.init();
//	std::string testServerAddr = "tcp://127.0.0.1:10204";
//	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
//	LunarEnv testEnv(testServerAddr, envName, num);
//	testEnv.init();
//	LOG4CXX_INFO(logger, "Env " << envName << " ready");

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
    option.totalStep = totalStep;
    option.targetUpdate = 512;
    option.statPathPrefix = "./prdqn_cart";
    option.rbCap = 4096;
    option.saveModel = false;
    option.exploreBegin = 1;
    option.exploreEnd = 0.1;
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.4;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1e-6;
    option.exploreStep = totalStep * 4 / 5;
    option.inputScale = 1;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.loadModel = false;

    RawPolicy policy(option.exploreBegin, outputNum);
    PrioRbDqnSingle<CartFcNet, LunarEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);
    dqn.train(batchSize);
//    dqn.save();
}

void test0(const int batchNum, const int totalStep) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
//	std::string testServerAddr = "tcp://127.0.0.1:10202";
//	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
//	AirEnv testEnv(testServerAddr, envName, num);
//	testEnv.init();

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
	torch::optim::RMSprop optimizer(model.parameters(),
	    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.totalStep = totalStep;
    option.targetUpdate = 512;
    option.statPathPrefix = "./prdqn_test0.txt";
    option.rbCap = 10240;
    option.saveModel = true;
    option.savePathPrefix = "./prdqn_test0";
    option.exploreBegin = 1;
    option.exploreEnd = 0.1;
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.4;
    option.pbBetaEnd = 0.4;
    option.pbEpsilon = 1e-6;
    option.exploreStep = totalStep * 4 / 5;
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;

    RawPolicy policy(option.exploreBegin, outputNum);

    PrioRbDqnSingle<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(batchNum);
}

void test1(const int batchNum, const int totalStep) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
//	std::string testServerAddr = "tcp://127.0.0.1:10202";
//	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
//	AirEnv testEnv(testServerAddr, envName, num);
//	testEnv.init();

	AirSKCnnNet model(outputNum);
	model.to(deviceType);
	AirSKCnnNet targetModel(outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
	torch::optim::RMSprop optimizer(model.parameters(),
	    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.totalStep = totalStep;
    option.targetUpdate = 512;
    option.statPathPrefix = "./prdqn_test1";
    option.rbCap = 32768;
    option.saveModel = true;
    option.savePathPrefix = "./prdqn_test1";
    option.exploreBegin = 1;
    option.exploreEnd = 0.9;
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.4;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1e-6;
    option.exploreStep = totalStep * 0.8;
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;

    RawPolicy policy(option.exploreBegin, outputNum);

    PrioRbDqnSingle<AirSKCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(batchNum);
}

void test2(const int batchNum, const int totalStep) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
//	std::string testServerAddr = "tcp://127.0.0.1:10202";
//	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
//	AirEnv testEnv(testServerAddr, envName, num);
//	testEnv.init();

	AirSKCnnNet model(outputNum);
	model.to(deviceType);
	AirSKCnnNet targetModel(outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
	torch::optim::RMSprop optimizer(model.parameters(),
	    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.totalStep = totalStep;
    option.targetUpdate = 512;
    option.statPathPrefix = "./prdqn_test2";
    option.rbCap = 32768;
    option.saveModel = true;
    option.savePathPrefix = "./prdqn_test2";
    option.exploreBegin = 0.9;
    option.exploreEnd = 0.8;
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.4;
    option.pbBetaEnd = 0.4;
    option.pbEpsilon = 1e-6;
    option.exploreStep = totalStep * 0.8;
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;

    RawPolicy policy(option.exploreBegin, outputNum);

    PrioRbDqnSingle<AirSKCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(batchNum);
}

//less steps to try
void test3(const int batchNum, const int totalStep) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
//	std::string testServerAddr = "tcp://127.0.0.1:10202";
//	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
//	AirEnv testEnv(testServerAddr, envName, num);
//	testEnv.init();

	AirSKCnnNet model(outputNum);
	model.to(deviceType);
	AirSKCnnNet targetModel(outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
	torch::optim::RMSprop optimizer(model.parameters(),
	    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.totalStep = totalStep;
    option.targetUpdate = 512;
    option.statPathPrefix = "./prdqn_test3";
    option.rbCap = 32768;
    option.saveModel = true;
    option.savePathPrefix = "./prdqn_test3";
    option.exploreBegin = 1;
    option.exploreEnd = 0.1;
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.4;
    option.pbBetaEnd = 0.4;
    option.pbEpsilon = 1e-6;
    option.exploreStep = totalStep * 0.8;
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;

    RawPolicy policy(option.exploreBegin, outputNum);

    PrioRbDqnSingle<AirSKCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(batchNum);
}

void test4(const int batchNum, const int totalStep) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
//	std::string testServerAddr = "tcp://127.0.0.1:10202";
//	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
//	AirEnv testEnv(testServerAddr, envName, num);
//	testEnv.init();

	AirSKCnnNet model(outputNum);
	model.to(deviceType);
	AirSKCnnNet targetModel(outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
	torch::optim::RMSprop optimizer(model.parameters(),
	    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.totalStep = totalStep;
    option.targetUpdate = 512;
    option.statPathPrefix = "./prdqn_test4";
    option.rbCap = 32768;
    option.saveModel = true;
    option.savePathPrefix = "./prdqn_test4";
    option.exploreBegin = 1;
    option.exploreEnd = 0.1;
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.4;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1e-6;
    option.exploreStep = totalStep * 0.5;
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/prdqn_test1";


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioRbDqnSingle<AirSKCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(batchNum);
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

//	testCartpole(atoi(argv[1]), atoi(argv[2]));
//	test0(atoi(argv[1]), atoi(argv[2]));
//	test1(atoi(argv[1]), atoi(argv[2]));
//	test2(atoi(argv[1]), atoi(argv[2]));
//	test3(atoi(argv[1]), atoi(argv[2]));
	test4(atoi(argv[1]), atoi(argv[2]));
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
//	test16(atoi(argv[1]), atoi(argv[2]));
//	test17(atoi(argv[1]), atoi(argv[2]));
//	test18(atoi(argv[1]), atoi(argv[2]));
//	test19(atoi(argv[1]), atoi(argv[2]));
//	test27(atoi(argv[1]), atoi(argv[2]));
//	testCal(atoi(argv[1]), atoi(argv[2]));

//	testtest14(atoi(argv[1]), atoi(argv[2]));

	LOG4CXX_INFO(logger, "End of test");
}
