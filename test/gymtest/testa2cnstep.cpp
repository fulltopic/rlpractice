/*
 * testa2cnstep.cpp
 *
 *  Created on: May 1, 2021
 *      Author: zf
 */




/*
 * testa2c.cpp
 *
 *  Created on: Apr 29, 2021
 *      Author: zf
 */


#include "alg/a2cnstep.hpp"

#include "gymtest/env/airenv.h"
#include "gymtest/env/lunarenv.h"
#include "gymtest/airnets/aircnnnet.h"
#include "gymtest/airnets/airacbmnet.h"
#include "gymtest/airnets/airacnet.h"
#include "gymtest/lunarnets/cartacnet.h"
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
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("testa2cn"));
const torch::Device deviceType = torch::kCUDA;

void test0(const int batchSize, const int epochNum) {
	const std::string envName = "CartPole-v0";
//	const std::string envName = "CartPoleNoFrameskip-v4";
	const int outputNum = 2;
	const int inputNum = 4;

	const int num = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, num);
	env.init();
//	std::string testServerAddr = "tcp://127.0.0.1:10204";
//	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
//	LunarEnv testEnv(testServerAddr, envName, num);
//	testEnv.init();
//	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartACFcNet model(inputNum, outputNum);
	model.to(deviceType);
//	CartACFcNet targetModel(inputNum, outputNum);
//	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4};
    DqnOption option(inputShape, deviceType);
//    option.gamma = 0.99;
//    option.batchSize = batchSize;
//    option.targetUpdate = 1024;
//    option.statPathPrefix = "./a2cn_test0";
//    option.rbCap = 10240;
//    option.saveModel = false;
//    option.savePathPrefix = "./a2cn_test0";
//    option.exploreBegin = 1;
//    option.exploreEnd = 0.1;
//    option.exploreStep = epochNum / 2;
//    option.loadModel = false;

    option.isAtari = false;
    option.donePerEp = 1;
    option.multiLifes = false;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.ppoLambda = 0.95;
    option.gamma = 0.99;
    option.statPathPrefix = "./a2cngae_test0";
    option.saveModel = true;
    option.savePathPrefix = "./a2cngae_test0";
    option.toTest = false;
    option.inputScale = 1;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.valueClip = false;
    option.normReward = false;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test17";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    A2CNStep<CartACFcNet, LunarEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum);
//    dqn.save();
}

void test1(const int batchSize, const int epochNum) {
	const std::string envName = "Breakout-v0";
//	const std::string envName = "Alien-v0";
//	const std::string envName = "BreakoutNoFrameskip-v4";
//	const std::string envName = "Pong-v0";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACCnnBmNet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 4096;
    option.batchSize = batchSize;
    option.entropyCoef = 0.001;
    option.statPathPrefix = "./a2cn_test1";
//    option.statPath = "./stat_a2cn_test1.txt";
//    option.teststatPath = "test_a2cn_test1";
    option.saveModel = true;
    option.savePathPrefix = "./a2cn_test1";
    option.loadModel = false;
    option.toTest = false;


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum);
//    dqn.save();
}

void test1_1(const int batchSize, const int epochNum) {
	const std::string envName = "Breakout-v0";
//	const std::string envName = "Alien-v0";
//	const std::string envName = "BreakoutNoFrameskip-v4";
//	const std::string envName = "Pong-v0";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACCnnBmNet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 4096;
    option.batchSize = batchSize;
    option.entropyCoef = 0.001;
    option.statPathPrefix = "./a2cn_test1_1";
//    option.statPath = "./stat_a2cn_test1.txt";
//    option.teststatPath = "test_a2cn_test1";
    option.saveModel = true;
    option.savePathPrefix = "./a2cn_test1_1";
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0510/a2cn_test1";
    option.loadModel = true;
    option.toTest = false;


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum);
//    dqn.save();
}

void test2(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v0";
	const std::string envName = "Alien-v0";
	const int outputNum = 18;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACCnnBmNet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 4096;
    option.batchSize = batchSize;
    option.entropyCoef = 0.001;
    option.statPathPrefix = "./a2cn_test2";
//    option.statPath = "./stat_a2cn_test2.txt";
//    option.teststatPath = "test_a2cn_test2";
    option.saveModel = false;
    option.savePathPrefix = "./a2cn_test2";
    option.loadModel = false;
    option.toTest = false;


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum);
//    dqn.save();
}

void test3(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v0";
	const std::string envName = "SpaceInvaders-v0";
	const int outputNum = 6;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACCnnBmNet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 4096;
    option.batchSize = batchSize;
    option.entropyCoef = 0.001;
    option.statPathPrefix = "./a2cn_test3";
//    option.statPath = "./stat_a2cn_test2.txt";
//    option.teststatPath = "test_a2cn_test2";
    option.saveModel = true;
    option.savePathPrefix = "./a2cn_test3";
    option.loadModel = false;
    option.toTest = false;


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum);
//    dqn.save();
}

void test4(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v0";
	const std::string envName = "SpaceInvaders-v0";
	const int outputNum = 6;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACCnnBmNet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 4096;
    option.batchSize = batchSize;
    option.entropyCoef = 0.001;
    option.statPathPrefix = "./a2cn_test4";
//    option.statPath = "./stat_a2cn_test2.txt";
//    option.teststatPath = "test_a2cn_test2";
    option.saveModel = false;
    option.savePathPrefix = "./a2cn_test4";
    option.loadModel = false;
    option.toTest = false;


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum);
//    dqn.save();
}

void test5(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v0";
	const std::string envName = "SpaceInvaders-v0";
	const int outputNum = 6;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACCnnBmNet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 4096;
    option.batchSize = batchSize;
    option.entropyCoef = 0.001;
    option.statPathPrefix = "./a2cn_test5";
//    option.statPath = "./stat_a2cn_test2.txt";
//    option.teststatPath = "test_a2cn_test2";
    option.saveModel = true;
    option.savePathPrefix = "./a2cn_test5";
    option.loadModel = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/a2cn_test3";
    option.toTest = false;


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum);
//    dqn.save();
}

void test6(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v0";
	const std::string envName = "SpaceInvaders-v0";
	const int outputNum = 6;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	const int testNum = 4;
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACCnnBmNet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 4096;
    option.batchSize = batchSize;
    option.entropyCoef = 0.001;
    option.statPathPrefix = "./a2cn_test6";
    option.saveModel = true;
    option.savePathPrefix = "./a2cn_test6";
    option.loadModel = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/a2cn_test5";
    option.toTest = true;
    option.testGapEp = 200;
    option.testBatch = testNum;


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, testEnv, policy, optimizer, maxStep, option);
    a2c.train(epochNum);
//    dqn.save();
}

//batchSize = 16
void test7(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v0";
	const std::string envName = "SpaceInvaders-v0";
	const int outputNum = 6;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	const int testNum = 4;
//	std::string testServerAddr = "tcp://127.0.0.1:10204";
//	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
//	AirEnv testEnv(testServerAddr, envName, testNum);
//	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACCnnBmNet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 4096;
    option.batchSize = batchSize;
    option.entropyCoef = 0.001;
    option.statPathPrefix = "./a2cn_test7";
    option.saveModel = true;
    option.savePathPrefix = "./a2cn_test7";
    option.loadModel = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/a2cn_test6";
    option.toTest = false;
    option.testGapEp = 200;
    option.testBatch = testNum;


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum);
//    dqn.save();
}


void test8(const int batchSize, const int epochNum) {
	const std::string envName = "Breakout-v0";
//	const std::string envName = "Alien-v0";
//	const std::string envName = "BreakoutNoFrameskip-v4";
//	const std::string envName = "Pong-v0";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACCnnNet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 4096;
    option.batchSize = batchSize;
    option.entropyCoef = 0.001;
    option.statPathPrefix = "./a2cn_test8";
//    option.statPath = "./stat_a2cn_test1.txt";
//    option.teststatPath = "test_a2cn_test1";
    option.saveModel = true;
    option.savePathPrefix = "./a2cn_test8";
    option.loadModel = false;
    option.toTest = false;


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum);
//    dqn.save();
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

	test0(atoi(argv[1]), atoi(argv[2]));
//	test1(atoi(argv[1]), atoi(argv[2]));
//	test1_1(atoi(argv[1]), atoi(argv[2]));
//	test2(atoi(argv[1]), atoi(argv[2]));
//	test3(atoi(argv[1]), atoi(argv[2]));
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
//	test16(atoi(argv[1]), atoi(argv[2]));
//	test17(atoi(argv[1]), atoi(argv[2]));
//	test18(atoi(argv[1]), atoi(argv[2]));
//	test19(atoi(argv[1]), atoi(argv[2]));
//	test27(atoi(argv[1]), atoi(argv[2]));
//	testCal(atoi(argv[1]), atoi(argv[2]));

	LOG4CXX_INFO(logger, "End of test");
}
