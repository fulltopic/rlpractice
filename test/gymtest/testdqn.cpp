/*
 * testdqn.cpp
 *
 *  Created on: Aug 17, 2021
 *      Author: zf
 */



#include "alg/dqn.hpp"

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


void test0(const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 1; //8
	const int outputNum = 2;
	const int inputNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartFcNet model(inputNum, outputNum);
	model.to(deviceType);
	CartFcNet targetModel(inputNum, outputNum);
	targetModel.to(deviceType);

    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
//    option.targetUpdate = 10000;
    option.targetUpdateStep = 2000;
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
    option.startStep = 1000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./dqn_test0";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test0";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    Dqn<CartFcNet, LunarEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}

void test00(const int epochNum) {
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

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
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
    //log
    option.statCap = 128;
    option.statPathPrefix = "./dqn_test00";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test00";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    Dqn<CartFcNet, ProbeEnvWrapper, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}

void test1(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdate = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 10240;
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
    option.batchSize = 32;
    option.startStep = 100;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./dqn_test1";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test1";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    Dqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(epochNum);
}


void test2(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdate = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 10240;
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
    option.startStep = 100;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./dqn_test2";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test2";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqn_test1";


    RawPolicy policy(option.exploreBegin, outputNum);

    Dqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);
    dqn.train(epochNum);
}

//TODO: inputScale = 256 would be better?
void test3(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdate = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 10240;
    //explore
    option.exploreBegin = 0.2;
    option.exploreEnd = 0.01;
    option.explorePart = 1;
    //input
    option.inputScale = 1;
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
    option.statCap = 128;
    option.statPathPrefix = "./dqn_test3";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test3";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqn_test2";


    RawPolicy policy(option.exploreBegin, outputNum);

    Dqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);
    dqn.train(epochNum);
}


void test4(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdate = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 10240;
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
    option.startStep = 100;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./dqn_test4";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test4";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqn_test2";


    RawPolicy policy(option.exploreBegin, outputNum);

    Dqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);
    dqn.train(epochNum);
}

void test5(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdate = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 10240;
    //explore
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
    option.startStep = 100;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./dqn_test1";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test1";
    option.loadModel = false;
    option.loadOptimizer = false;


    RawPolicy policy(option.exploreBegin, outputNum);

    Dqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

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

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
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
    //log
    option.statCap = 128;
    option.statPathPrefix = "./dqn_testprobe";
    //model
    option.saveModel = false;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqn_testprobe";



    RawPolicy policy(option.exploreBegin, outputNum);

    Dqn<CartFcNet, ProbeEnvWrapper, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);
    dqn.test(epochNum);
}

void testCart(const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 1; //8
	const int outputNum = 2;
	const int inputNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartFcNet model(inputNum, outputNum);
	model.to(deviceType);
//	CartFcNet targetModel(inputNum, outputNum);
//	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
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
    //log
    option.statCap = 128;
    option.statPathPrefix = "./dqn_testcart";
    //model
    option.saveModel = false;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqn_test0";



    RawPolicy policy(option.exploreBegin, outputNum);

    Dqn<CartFcNet, LunarEnv, RawPolicy, torch::optim::RMSprop> dqn(model, model, env, env, policy, optimizer, option);
    dqn.test(epochNum);
}

void testPong(const int epochNum) {
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


//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdate = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 128;
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
    option.startStep = 100;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./dqn_testtest4";
    //model
    option.saveModel = false;
    option.savePathPrefix = "./dqn_testtest4";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqn_test4";


    RawPolicy policy(option.exploreBegin, outputNum);

    Dqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, model, env, env, policy, optimizer, option);
    dqn.test(epochNum);
}


void testBreakout(const int epochNum) {
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


//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
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
    option.statCap = 128;
    option.statPathPrefix = "./dqn_testtest121";
    //model
    option.saveModel = false;
    option.savePathPrefix = "./dqn_testtest121";
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqn_test121";


    RawPolicy policy(option.exploreBegin, outputNum);

    Dqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, model, env, env, policy, optimizer, option);
    dqn.test(epochNum);
}

void test100(const int epochNum) {
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
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 10000;
    option.tau = 1;
    //buffer
    option.rbCap = 20480;
    //explore
    option.exploreBegin = 1;
    option.exploreEnd = 0.7;
    option.explorePart = 1;
    //input
    option.inputScale = 256;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./dqn_test100";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test100";
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqn_test2";


    RawPolicy policy(option.exploreBegin, outputNum);

    Dqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);
    dqn.train(epochNum);
}

void test101(const int epochNum) {
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
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdate = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 20480;
    //explore
    option.exploreBegin = 0.9;
    option.exploreEnd = 0.8;
    option.explorePart = 1;
    //input
    option.inputScale = 256;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./dqn_test101";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test101";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqn_test5";


    RawPolicy policy(option.exploreBegin, outputNum);

    Dqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);
    dqn.train(epochNum);
}

void test103(const int epochNum) {
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
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdate = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 20480;
    //explore
    option.exploreBegin = 0.8;
    option.exploreEnd = 0.7;
    option.explorePart = 1;
    //input
    option.inputScale = 256;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./dqn_test103";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test103";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqn_test102";


    RawPolicy policy(option.exploreBegin, outputNum);

    Dqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);
    dqn.train(epochNum);
}


void test104(const int epochNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
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
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdate = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 20480;
    //explore
    option.exploreBegin = 0.7;
    option.exploreEnd = 0.6;
    option.explorePart = 1;
    //input
    option.inputScale = 256;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./dqn_test104";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test104";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqn_test103";


    RawPolicy policy(option.exploreBegin, outputNum);

    Dqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);
    dqn.train(epochNum);
}

void test105(const int epochNum) {
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
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001).eps(0.01).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdate = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 20480;
    //explore
    option.exploreBegin = 0.6;
    option.exploreEnd = 0.5;
    option.explorePart = 1;
    //input
    option.inputScale = 256;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./dqn_test105";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test105";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqn_test104";


    RawPolicy policy(option.exploreBegin, outputNum);

    Dqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);
    dqn.train(epochNum);
}
void test102(const int epochNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
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
    option.targetUpdate = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 20480;
    //explore
    option.exploreBegin = 1;
    option.exploreEnd = 0.1;
    option.explorePart = 0.1; //abs = 25,000, step = 250,000
    //input
    option.inputScale = 256;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./dqn_test102";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test102";
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqn_test5";


    RawPolicy policy(option.exploreBegin, outputNum);

    Dqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, env, policy, optimizer, option);
    dqn.train(epochNum);
}

void test120(const int epochNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
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
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00001));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 10000;
    option.tau = 1;
    //buffer
    option.rbCap = 20480;
    //explore
    option.exploreBegin = 1;
    option.exploreEnd = 0.7;
    option.explorePart = 1;
    //input
    option.inputScale = 256;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./dqn_test120";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test120";
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqn_test2";


    RawPolicy policy(option.exploreBegin, outputNum);

    Dqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);
    dqn.train(epochNum);
}

void test121(const int epochNum) {
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
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00001));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 10000;
    option.tau = 1;
    //buffer
    option.rbCap = 20480;
    //explore
    option.exploreBegin = 0.7;
    option.exploreEnd = 0.5;
    option.explorePart = 1;
    //input
    option.inputScale = 256;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./dqn_test121";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test121";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqn_test120";


    RawPolicy policy(option.exploreBegin, outputNum);

    Dqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);
    dqn.train(epochNum);
}

void test122(const int epochNum) {
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
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.0001));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
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
    option.exploreBegin = 0.5;
    option.exploreEnd = 0.1;
    option.explorePart = 1;
    //input
    option.inputScale = 256;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1;
    //log
    option.logInterval = 1000;
    option.statCap = 128;
    option.statPathPrefix = "./dqn_test122";
    //model
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test122";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqn_test121";


    RawPolicy policy(option.exploreBegin, outputNum);

    Dqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);
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

//	test0(atoi(argv[1]));
	test122(atoi(argv[1]));
//	testPong(atoi(argv[1]));
//	testBreakout(atoi(argv[1]));


	LOG4CXX_INFO(logger, "End of test");
}
