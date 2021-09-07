/*
 * testnoisydqn.cpp
 *
 *  Created on: Aug 31, 2021
 *      Author: zf
 */





#include "alg/noisydoubledqn.hpp"
#include "alg/noisydqn.hpp"

#include "gymtest/env/airenv.h"
#include "gymtest/env/lunarenv.h"
#include "gymtest/airnets/aircnnnet.h"
#include "gymtest/airnets/airacbmnet.h"
#include "gymtest/airnets/airacnet.h"
#include "gymtest/airnets/airdueling.h"
#include "gymtest/airnets/airacbmsmallkernelnet.h"
#include "gymtest/lunarnets/cartacnet.h"
#include "gymtest/lunarnets/cartnet.h"
#include "gymtest/lunarnets/cartduelnet.h"
#include "gymtest/airnets/airachonet.h"
#include "gymtest/noisynets/noisycartfcnet.h"
#include "gymtest/noisynets/noisyaircnnnet.h"
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
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("noisydqntest"));
const torch::Device deviceType = torch::kCUDA;

void testProbe(const int epochNum) {
	const int batchSize = 1;
	const int inputNum = 4;
	const int envId = 5;
	const int outputNum = 2;

	const int envNum = batchSize;
	ProbeEnvWrapper env(inputNum, envId, envNum);

	NoisyCartFcNet model(inputNum, outputNum);
	model.to(deviceType);
	NoisyCartFcNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{envNum, 4};
    DqnOption option(inputShape, deviceType);
    option.envNum = envNum;
    //target model
    option.targetUpdateStep = 100;
    option.tau = 1;
    //buffer
    option.rbCap = 100;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 1;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 4;
    option.startStep = 50;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./noisydqn_testprobe";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./noisydqn_testprobe";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    NoisyDoubleDqn<NoisyCartFcNet, ProbeEnvWrapper, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);
    dqn.train(epochNum);
}

void testDqnProbe(const int epochNum) {
	const int batchSize = 1;
	const int inputNum = 4;
	const int envId = 5;
	const int outputNum = 2;

	const int envNum = batchSize;
	ProbeEnvWrapper env(inputNum, envId, envNum);

	NoisyCartFcNet model(inputNum, outputNum);
	model.to(deviceType);
	NoisyCartFcNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{envNum, 4};
    DqnOption option(inputShape, deviceType);
    option.envNum = envNum;
    //target model
    option.targetUpdateStep = 100;
    option.tau = 1;
    //buffer
    option.rbCap = 100;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 1;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 4;
    option.startStep = 50;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./noisydqn_testprobe";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./noisydqn_testprobe";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    NoisyDqn<NoisyCartFcNet, ProbeEnvWrapper, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);
    dqn.train(epochNum);
}

void testCart(const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 1; //8
	const int outputNum = 2;
	const int inputNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	NoisyCartFcNet model(inputNum, outputNum);
	model.to(deviceType);
	NoisyCartFcNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);

//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 10000;
    option.tau = 1;
    //buffer
    option.rbCap = 10000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 1;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 4;
    option.startStep = 1000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./noisydoubledqn_testcart";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./noisydoubledqn_testcart";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    NoisyDoubleDqn<NoisyCartFcNet, LunarEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}

void testCartV2(const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 1; //8
	const int outputNum = 2;
	const int inputNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	NoisyCartFcNet model(inputNum, outputNum);
	model.to(deviceType);
	NoisyCartFcNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);

//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 10000;
    option.tau = 1;
    //buffer
    option.rbCap = 10000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 1;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 4;
    option.startStep = 1000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./noisydoubledqnv2_testcart";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./noisydoubledqnv2_testcart";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    NoisyDoubleDqn<NoisyCartFcNet, LunarEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}

void testDqnCart(const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 1; //8
	const int outputNum = 2;
	const int inputNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	NoisyCartFcNet model(inputNum, outputNum);
	model.to(deviceType);
	NoisyCartFcNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);

//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 10000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 1;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 4;
    option.startStep = 1000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./noisydqn_testcart";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./noisydqn_testcart";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    NoisyDqn<NoisyCartFcNet, LunarEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}


void testtestCart(const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 1; //8
	const int outputNum = 2;
	const int inputNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	NoisyCartFcNet model(inputNum, outputNum);
	model.to(deviceType);
//	CartFcNet targetModel(inputNum, outputNum);
//	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    option.targetUpdate = 100;
    option.tau = 1;
    //buffer
    option.rbCap = 100;
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
    //log
    option.statCap = 128;
    option.statPathPrefix = "./noisydqn_testtestcart";
    //model
    option.saveModel = false;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/noisydqn_testcart";



    RawPolicy policy(option.exploreBegin, outputNum);

    NoisyDoubleDqn<NoisyCartFcNet, LunarEnv, RawPolicy, torch::optim::Adam> dqn(model, model, env, env, policy, optimizer, option);
    dqn.test(epochNum);
}


void testPong1(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	NoisyAirCnnNet model(outputNum);
	model.to(deviceType);
	NoisyAirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 100000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 256;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 128;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./noisydoubledqn_testpong1";
    //model
    option.saveThreshold = -20;
    option.saveStep = 1;
    option.saveModel = true;
    option.savePathPrefix = "./noisydoubledqn_testpong1";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    NoisyDoubleDqn<NoisyAirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}

//TODO: try smaller lr

void testPongLog(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	NoisyAirCnnNet model(outputNum);
	model.to(deviceType);
	NoisyAirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 10000;
    option.tau = 1;
    //buffer
    option.rbCap = 10240;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 256;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 128;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./noisydoubledqn_testponglog";
    //model
    option.saveThreshold = -19;
    option.saveStep = 1;
    option.saveModel = false;
    option.savePathPrefix = "./noisydoubledqn_testponglog";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    NoisyDoubleDqn<NoisyAirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}


void testPong2(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	NoisyAirCnnNet model(outputNum);
	model.to(deviceType);
	NoisyAirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 10240;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 256;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 128;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./noisydoubledqn_testpong2";
    //model
    option.saveThreshold = -20;
    option.saveStep = 1;
    option.saveModel = true;
    option.savePathPrefix = "./noisydoubledqn_testpong2";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    NoisyDoubleDqn<NoisyAirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}


void testPong3(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	NoisyAirCnnNet model(outputNum);
	model.to(deviceType);
	NoisyAirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 1024;
    option.tau = 1;
    //buffer
    option.rbCap = 100000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 256;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 32;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./noisydoubledqn_testpong3";
    //model
    option.saveThreshold = -20;
    option.saveStep = 1;
    option.saveModel = true;
    option.savePathPrefix = "./noisydoubledqn_testpong3";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    NoisyDoubleDqn<NoisyAirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

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
//    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getError());
    log4cxx::LogManager::getLoggerRepository()->setConfigured(true);
}
}

int main(int argc, char** argv) {
	logConfigure(false);

//	testtestCart(atoi(argv[1]));
//	test103(atoi(argv[1]));
	testPong1(atoi(argv[1]));
//	testBreakout(atoi(argv[1]));

//	testCartV2(atoi(argv[1]));
//	testProbe(atoi(argv[1]));
//	testPong3(atoi(argv[1]));

//	testDqnProbe(atoi(argv[1]));
//	testDqnCart(atoi(argv[1]));


//	testPongLog(atoi(argv[1]));

//	testtestCart(atoi(argv[1]));

	LOG4CXX_INFO(logger, "End of test");
}