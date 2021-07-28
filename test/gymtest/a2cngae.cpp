/*
 * a2cngae.cpp
 *
 *  Created on: Jul 10, 2021
 *      Author: zf
 */



#include "alg/a2cnstepnorm.hpp"
#include "alg/a2cnstepgae.hpp"

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
#include "alg/dqnoption.h"


#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/simplelayout.h>
#include <log4cxx/logmanager.h>

#include <vector>

namespace {
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("a2cngae"));
const torch::Device deviceType = torch::kCUDA;

void test0(const int epochNum) {
	const int batchSize = 50;
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

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    A2CNStepGae<CartACFcNet, LunarEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum);
}

void test1(const int epochNum) {
	const int batchSize = 50;

	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./a2cngae_test1";
    option.saveModel = true;
    option.savePathPrefix = "./a2cngae_test1";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test9";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    A2CNStepGae<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum);
}

//TODO: Since rewards are < 10 or > 100, could be clipped into 10 or 20

void test2(const int epochNum) {
	const int batchSize = 50;

	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./a2cngae_test2";
    option.saveModel = true;
    option.savePathPrefix = "./a2cngae_test2";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/a2cngae_test1";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    A2CNStepGae<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum);
}


void test3(const int epochNum) {
	const int batchSize = 50;

	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./a2cngae_test3";
    option.saveModel = true;
    option.savePathPrefix = "./a2cngae_test3";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/a2cngae_test2";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    A2CNStepGae<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum);
}


void test4(const int epochNum) {
	const int batchSize = 50;

	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./a2cngae_test4";
    option.saveModel = true;
    option.savePathPrefix = "./a2cngae_test4";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/a2cngae_test3";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    A2CNStepGae<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum);
}




void test5(const int epochNum) {
	const int batchSize = 50;

	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./a2cngae_test5";
    option.saveModel = true;
    option.savePathPrefix = "./a2cngae_test5";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.normReward = false;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/a2cngae_test3";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 20;
    //TODO: testenv
    A2CNStepGae<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
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

	test5(atoi(argv[1]));
//	test18(atoi(argv[1]));

//Test
//	testtest204(atoi(argv[1]));

//GAE
//	test102(atoi(argv[1]));

//Clipped
//	test205(atoi(argv[1]));

//	testtest0(atoi(argv[1]), atoi(argv[2]));

	LOG4CXX_INFO(logger, "End of test");
}
