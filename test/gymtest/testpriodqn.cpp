/*
 * testpriodqn.cpp
 *
 *  Created on: Aug 26, 2021
 *      Author: zf
 */



#include "alg/priodqn.hpp"

#include "gymtest/env/airenv.h"
#include "gymtest/env/lunarenv.h"
#include "gymtest/airnets/aircnnnet.h"
#include "gymtest/airnets/airacbmnet.h"
#include "gymtest/airnets/airacnet.h"
#include "gymtest/airnets/airacbmsmallkernelnet.h"
#include "gymtest/lunarnets/cartacnet.h"
#include "gymtest/lunarnets/cartnet.h"
#include "gymtest/lunarnets/cartqnet.h"
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


void testProbe(const int epochNum) {
	const int batchSize = 1;
	const int inputNum = 4;
	const int envId = 5;
	const int outputNum = 2;

	const int envNum = batchSize;
	ProbeEnvWrapper env(inputNum, envId, envNum);

	CartFcQNet model(inputNum, outputNum);
	model.to(deviceType);
	CartFcQNet targetModel(inputNum, outputNum);
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
    //prio
    option.rbCap = 1024; //Cap to be 2^x
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.4;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1e-6;
    option.pbBetaPart = 1;
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
    //log
    option.statCap = 128;
    option.statPathPrefix = "./priodqn_testprobe";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./priodqn_testprobe";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<CartFcQNet, ProbeEnvWrapper, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);
    dqn.train(epochNum);
}


void testCart(const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 1; //8
	const int outputNum = 2;
	const int inputNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartFcQNet model(inputNum, outputNum);
	model.to(deviceType);
	CartFcQNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);

    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025));//.eps(0.001).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 1000;
    option.tau = 1;
    //prio
    option.rbCap = 8192; //Cap to be 2^x
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.4;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1;
    option.pbBetaPart = 0.9;
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
    option.batchSize = 16;
    option.startStep = 1000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./priodqn_testcart";
    //model
    option.saveModel = false;
    option.savePathPrefix = "./priodqn_testcart";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<CartFcQNet, LunarEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}


void testCartLog(const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 1; //8
	const int outputNum = 2;
	const int inputNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartFcQNet model(inputNum, outputNum);
	model.to(deviceType);
	CartFcQNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);

    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025));//.eps(0.001).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 2000;
    option.tau = 1;
    //prio
    option.rbCap = 32; //Cap to be 2^x
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.4;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1e-6;
    option.pbBetaPart = 0.9;
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
    option.startStep = 32;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./priodqn_testcartlog";
    //model
    option.saveModel = false;
    option.savePathPrefix = "./priodqn_testcartlog";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<CartFcQNet, LunarEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}



void testCart1(const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 1; //8
	const int outputNum = 2;
	const int inputNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartFcQNet model(inputNum, outputNum);
	model.to(deviceType);
	CartFcQNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);

    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.001).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 2000;
    option.tau = 1;
    //prio
    option.rbCap = 8192; //Cap to be 2^x
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.4;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1e-6;
    option.pbBetaPart = 0.9;
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
    option.batchSize = 16;
    option.startStep = 1000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./priodqn_testcart1";
    //model
    option.saveModel = false;
    option.savePathPrefix = "./priodqn_testcart1";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<CartFcQNet, LunarEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}

void testCart2(const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 1; //8
	const int outputNum = 2;
	const int inputNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartFcQNet model(inputNum, outputNum);
	model.to(deviceType);
	CartFcQNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);

//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.001).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 2000;
    option.tau = 1;
    //prio
    option.rbCap = 8192; //Cap to be 2^x
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.4;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1e-6;
    option.pbBetaPart = 0.9;
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
    option.batchSize = 16;
    option.startStep = 1000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./priodqn_testcart2";
    //model
    option.saveModel = false;
    option.savePathPrefix = "./priodqn_testcart2";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<CartFcQNet, LunarEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}

void testCart3(const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 1; //8
	const int outputNum = 2;
	const int inputNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartFcQNet model(inputNum, outputNum);
	model.to(deviceType);
	CartFcQNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);

//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.001).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 2000;
    option.tau = 1;
    //prio
    option.rbCap = 8192; //Cap to be 2^x
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.4;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1e-6;
    option.pbBetaPart = 0.9;
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
    option.batchSize = 16;
    option.startStep = 1000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./priodqn_testcart3";
    //model
    option.saveModel = false;
    option.savePathPrefix = "./priodqn_testcart3";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<CartFcQNet, LunarEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}

void testPong0(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 1000;
    option.tau = 1;
    //prio
    option.rbCap = 8192 * 8; //Cap to be 2^x
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.4;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1e-3;
    option.pbBetaPart = 0.9;    //explore
    option.exploreBegin = 1;
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
    option.startStep = 8192;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./priodqn_testpong0";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./priodqn_testpong0";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
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

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
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
    option.targetUpdateStep = 5000;
    option.tau = 1;
    //prio
    option.rbCap = 8192 * 4; //Cap to be 2^x
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 1;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1e-6;
    option.pbBetaPart = 1;    //explore
    option.exploreBegin = 0.1;
    option.exploreEnd = 0.1;
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
    option.statPathPrefix = "./priodqn_testpong1";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./priodqn_testpong1";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/priodqn_testpong0";


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);

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

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
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
    option.targetUpdateStep = 5000;
    option.tau = 1;
    //prio
    option.rbCap = 8192 * 4; //Cap to be 2^x
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.8;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1e-6;
    option.pbBetaPart = 1;    //explore
    option.exploreBegin = 0.1;
    option.exploreEnd = 0.1;
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
    option.statPathPrefix = "./priodqn_testpong2";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./priodqn_testpong2";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/priodqn_testpong1";


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);

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

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
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
    option.targetUpdateStep = 5000;
    option.tau = 1;
    //prio
    option.rbCap = 8192 * 4; //Cap to be 2^x
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 1;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1e-6;
    option.pbBetaPart = 1;    //explore
    option.exploreBegin = 0.1;
    option.exploreEnd = 0.1;
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
    option.statPathPrefix = "./priodqn_testpong3";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./priodqn_testpong3";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/priodqn_testpong1";


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}

void testPong4(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
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
    option.targetUpdateStep = 5000;
    option.tau = 1;
    //prio
    option.rbCap = 8192 * 4; //Cap to be 2^x
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.8;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1e-6;
    option.pbBetaPart = 1;    //explore
    option.exploreBegin = 0.1;
    option.exploreEnd = 0.1;
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
    option.statPathPrefix = "./priodqn_testpong4";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./priodqn_testpong4";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/priodqn_testpong2";


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}

void testPong5(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
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
    option.targetUpdateStep = 5000;
    option.tau = 1;
    //prio
    option.rbCap = 8192 * 4; //Cap to be 2^x
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.5;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1e-6;
    option.pbBetaPart = 1;    //explore
    option.exploreBegin = 1;
    option.exploreEnd = 0.01;
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
    option.statPathPrefix = "./priodqn_testpong5";
    //model
    option.saveThreshold = -14;
    option.saveStep = 1;
    option.saveModel = true;
    option.savePathPrefix = "./priodqn_testpong5";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/priodqn_testpong1";


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}


void testPong6(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
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
    option.targetUpdateStep = 5000;
    option.tau = 1;
    //prio
    option.rbCap = 8192 * 4; //Cap to be 2^x
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.6;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1e-6;
    option.pbBetaPart = 1;    //explore
    option.exploreBegin = 0.2;
    option.exploreEnd = 0.1;
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
    option.statPathPrefix = "./priodqn_testpong6";
    //model
    option.saveThreshold = -14;
    option.saveStep = 1;
    option.saveModel = true;
    option.savePathPrefix = "./priodqn_testpong6";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/priodqn_testpong2";


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}

void testPong7(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
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
    option.targetUpdateStep = 5000;
    option.tau = 1;
    //prio
    option.rbCap = 8192 * 4; //Cap to be 2^x
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.6;
    option.pbBetaEnd = 1;
    option.pbEpsilon = 1e-6;
    option.pbBetaPart = 1;    //explore
    option.exploreBegin = 0.2;
    option.exploreEnd = 0.1;
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
    option.statPathPrefix = "./priodqn_testpong7";
    //model
    option.saveThreshold = -14;
    option.saveStep = 1;
    option.saveModel = true;
    option.savePathPrefix = "./priodqn_testpong7";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/priodqn_testpong6";


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}


void testPong8(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
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
    option.targetUpdateStep = 5000;
    option.tau = 1;
    //prio
    option.rbCap = 8192 * 4; //Cap to be 2^x
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.9;
    option.pbBetaEnd = 9;
    option.pbEpsilon = 1e-6;
    option.pbBetaPart = 1;    //explore
    option.exploreBegin = 0.1;
    option.exploreEnd = 0.1;
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
    option.statPathPrefix = "./priodqn_testpong8";
    //model
    option.saveThreshold = -5;
    option.saveStep = 1;
    option.saveModel = true;
    option.savePathPrefix = "./priodqn_testpong8";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/priodqn_testpong7";


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}


void testPong9(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
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
    option.targetUpdateStep = 5000;
    option.tau = 1;
    //prio
    option.rbCap = 8192 * 4; //Cap to be 2^x
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.9;
    option.pbBetaEnd = 0.9;
    option.pbEpsilon = 1e-6;
    option.pbBetaPart = 1;    //explore
    option.exploreBegin = 0.1;
    option.exploreEnd = 0.1;
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
    option.statPathPrefix = "./priodqn_testpong9";
    //model
    option.saveThreshold = -5;
    option.saveStep = 1;
    option.saveModel = true;
    option.savePathPrefix = "./priodqn_testpong9";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/priodqn_testpong7";


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}



void testPong10(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
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
    option.targetUpdateStep = 5000;
    option.tau = 1;
    //prio
    option.rbCap = 8192 * 4; //Cap to be 2^x
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.9;
    option.pbBetaEnd = 0.9;
    option.pbEpsilon = 1e-6;
    option.pbBetaPart = 1;    //explore
    option.exploreBegin = 0.1;
    option.exploreEnd = 0.1;
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
    option.statPathPrefix = "./priodqn_testpong10";
    //model
    option.saveThreshold = 4;
    option.saveStep = 1;
    option.saveModel = true;
    option.savePathPrefix = "./priodqn_testpong10";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/priodqn_testpong9";


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}



void testPong11(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
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
    option.targetUpdateStep = 5000;
    option.tau = 1;
    //prio
    option.rbCap = 8192 * 4; //Cap to be 2^x
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.9;
    option.pbBetaEnd = 0.9;
    option.pbEpsilon = 1e-6;
    option.pbBetaPart = 1;    //explore
    option.exploreBegin = 0.1;
    option.exploreEnd = 0.05;
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
    option.statPathPrefix = "./priodqn_testpong11";
    //model
    option.saveThreshold = 4;
    option.saveStep = 1;
    option.saveModel = true;
    option.savePathPrefix = "./priodqn_testpong11";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/priodqn_testpong9";


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}

void testLog(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
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
    option.targetUpdateStep = 5000;
    option.tau = 1;
    //prio
    option.rbCap = 1024; //Cap to be 2^x
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.9;
    option.pbBetaEnd = 0.9;
    option.pbEpsilon = 1e-6;
    option.pbBetaPart = 1;    //explore
    option.exploreBegin = 0.1;
    option.exploreEnd = 0.05;
    option.explorePart = 1;
    //input
    option.inputScale = 256;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 4;
    option.startStep = 100;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./priodqn_testponglog";
    //model
    option.saveThreshold = 4;
    option.saveStep = 1;
    option.saveModel = true;
    option.savePathPrefix = "./priodqn_testponglog";
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/priodqn_testpong9";


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}

void testtestPong(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
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
    option.targetUpdateStep = 5000;
    option.tau = 1;
    //prio
    option.rbCap = 8192 * 4; //Cap to be 2^x
    option.pbAlpha = 0.6;
    option.pbBetaBegin = 0.9;
    option.pbBetaEnd = 0.9;
    option.pbEpsilon = 1e-6;
    option.pbBetaPart = 1;    //explore
    option.exploreBegin = 0.1;
    option.exploreEnd = 0.1;
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
    option.statPathPrefix = "./priodqn_testtestpong";
    //model
    option.saveThreshold = 4;
    option.saveStep = 1;
    option.saveModel = false;
    option.savePathPrefix = "./priodqn_testtestpong";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/priodqn_testpong10";


    RawPolicy policy(option.exploreBegin, outputNum);

    PrioDqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);
    dqn.test(epochNum);
}
//TODO: increase beta
//TODO: sometimes, beta > 1 makes better performance, but beta is not encouraged to > 1
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
	testPong0(atoi(argv[1]));
//	testBreakout(atoi(argv[1]));

//	testCart(atoi(argv[1]));
//	testProbe(atoi(argv[1]));
//	testPong(atoi(argv[1]));
//	testtestPong(atoi(argv[1]));
//	testCartLog(atoi(argv[1]));

	LOG4CXX_INFO(logger, "End of test");
}
