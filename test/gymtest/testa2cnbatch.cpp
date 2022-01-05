/*
 * testa2cnbatch.cpp
 *
 *  Created on: May 11, 2021
 *      Author: zf
 */



#include "alg/a2cnstep.hpp"
#include "alg/a2cnstepnorm.hpp"

#include "gymtest/env/airenv.h"
#include "gymtest/env/lunarenv.h"
#include "gymtest/airnets/aircnnnet.h"
#include "gymtest/airnets/airacbmnet.h"
#include "gymtest/airnets/airacnet.h"
#include "gymtest/airnets/airacbmsmallkernelnet.h"
#include "gymtest/airnets/airachonet.h"
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
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("testa2cnbatch"));
const torch::Device deviceType = torch::kCUDA;

void test0(const int batchSize, const int epochNum) {
	const std::string envName = "CartPole-v0";
//	const std::string envName = "CartPoleNoFrameskip-v4";
	const int outputNum = 2;
	const int inputNum = 4;

	const int num = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
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
    option.gamma = 0.99;
    option.batchSize = batchSize;
    option.targetUpdate = 1024;
    option.statPathPrefix = "./a2cnbatch_test0";
    option.saveModel = false;
    option.loadModel = false;
    option.inputScale = 1;
    option.isAtari = false;

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    A2CNStep<CartACFcNet, LunarEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, false);
//    dqn.save();
}

void test1(const int batchSize, const int epochNum) {
	const std::string envName = "Breakout-v0";
//	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
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
    option.entropyCoef = 0.001;
    option.statPathPrefix = "./a2cnbatch_test1";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test1";
    option.loadModel = false;
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
//    dqn.save();
}

void test2(const int batchSize, const int epochNum) {
	const std::string envName = "Breakout-v0";
//	const std::string envName = "BreakoutNoFrameskip-v4";
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
    option.entropyCoef = 0.001;
    option.statPathPrefix = "./a2cnbatch_test2";
    option.saveModel = false;
    option.savePathPrefix = "./a2cnbatch_test2";
    option.loadModel = false;
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 4;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
//    dqn.save();
}

void test3(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v0";
	const std::string envName = "Alien-v0";
	const int outputNum = 18;
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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.entropyCoef = 0.001;
    option.statPathPrefix = "./a2cnbatch_test3";
    option.saveModel = false;
    option.savePathPrefix = "./a2cnbatch_test3";
    option.loadModel = false;
    option.toTest = false;
    option.isAtari = true;
    option.batchSize = batchSize;
    option.rewardScale = 100;
    option.rewardMin = -10;
    option.rewardMax = 10;

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum,  true);
//    dqn.save();
}

void test4(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACSKCnnBmNet model(outputNum);
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
    option.entropyCoef = 0.001;
    option.statPathPrefix = "./a2cnbatch_test4";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test4";
    option.loadModel = false;
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -10;
    option.rewardMax = 10;


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACSKCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
//    dqn.save();
}

void test5(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
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
    option.entropyCoef = 0.001;
    option.statPathPrefix = "./a2cnbatch_test5";
    option.saveModel = false;
    option.loadModel = false;
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -10;
    option.rewardMax = 10;


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 16;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, false);
//    dqn.save();
}

void test6(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
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
    option.entropyCoef = 0.001;
    option.statPathPrefix = "./a2cnbatch_test5";
    option.saveModel = false;
    option.savePathPrefix = "./a2cnbatch_test6";
    option.loadModel = false;
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -10;
    option.rewardMax = 10;


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
//    dqn.save();
}

void test7(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACSKCnnBmNet model(outputNum);
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
    option.entropyCoef = 0.001;
    option.statPathPrefix = "./a2cnbatch_test7";
    option.saveModel = false;
    option.savePathPrefix = "./a2cnbatch_test7";
    option.loadModel = false;
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -10;
    option.rewardMax = 10;


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStepNorm<AirACSKCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum);
//    dqn.save();
}

void test8(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.entropyCoef = 0.001;
    option.statPathPrefix = "./a2cnbatch_test8";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test8";
    option.loadModel = false;
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -10;
    option.rewardMax = 10;


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStepNorm<AirACCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum);
//    dqn.save();
}

//Maybe too small rewards are, makes value/act loss slim and encourage huge entropy. And the trajectory failed to escape trap
void test9(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "AlienNoFrameskip-v4";
//	const int outputNum = 4; //Error in output number
	const int outputNum = 18;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACSKCnnBmNet model(outputNum);
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
    option.entropyCoef = 0.001;
    option.statPathPrefix = "./a2cnbatch_test9";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test9";
    option.loadModel = false;
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 100;
    option.rewardMin = -10;
    option.rewardMax = 10;


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACSKCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
//    dqn.save();
}

//Maybe too small rewards are, makes value/act loss slim and encourage huge entropy. And the trajectory failed to escape trap
void test10(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACSKCnnBmNet model(outputNum);
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
    option.entropyCoef = 0.001;
    option.statPathPrefix = "./a2cnbatch_test10";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test10";
    option.loadModel = false;
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACSKCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
//    dqn.save();
}

void test11(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACSKCnnBmNet model(outputNum);
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
    option.entropyCoef = 0.001;
    option.statPathPrefix = "./a2cnbatch_test11";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test11";
    option.loadModel = false;
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/a2cnbatch_test10";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACSKCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
//    dqn.save();
}

void test12(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACSKCnnBmNet model(outputNum);
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
    option.entropyCoef = 0.01;
    option.statPathPrefix = "./a2cnbatch_test12";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test12";
    option.loadModel = false;
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/a2cnbatch_test10";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACSKCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
//    dqn.save();
}

void test13(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACSKCnnBmNet model(outputNum);
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
    option.entropyCoef = 0.01;
    option.statPathPrefix = "./a2cnbatch_test13";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test13";
    option.loadModel = false;
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/a2cnbatch_test12";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACSKCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
//    dqn.save();
}

void test14(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACSKCnnBmNet model(outputNum);
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
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.statPathPrefix = "./a2cnbatch_test14";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test14";
    option.loadModel = false;
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/a2cnbatch_test13";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACSKCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
//    dqn.save();
}

void testtest14(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACSKCnnBmNet model(outputNum);
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
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.statPathPrefix = "./a2cnbatch_testtest14";
    option.saveModel = false;
    option.savePathPrefix = "./a2cnbatch_testtest14";
    option.loadModel = false;
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/a2cnbatch_test14";

    option.toTest = true;

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACSKCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
//    a2c.train(epochNum, true);
    a2c.load();
    a2c.test(batchSize, epochNum);
//    dqn.save();
}


void test15(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACSKCnnBmNet model(outputNum);
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
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.statPathPrefix = "./a2cnbatch_test15";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test15";
    option.loadModel = false;
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = 0;
    option.rewardMax = 10;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/a2cnbatch_test13";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACSKCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
//    dqn.save();
}

void test16(const int batchSize, const int epochNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACSKCnnBmNet model(outputNum);
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
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.statPathPrefix = "./a2cnbatch_test16";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test16";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/a2cnbatch_test14";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 5;
    //TODO: testenv
    A2CNStep<AirACSKCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}

void test17(const int batchSize, const int epochNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
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
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.statPathPrefix = "./a2cnbatch_test17";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test17";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 5;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}

void test18(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACSKCnnBmNet model(outputNum);
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
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.statPathPrefix = "./a2cnbatch_test18";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test18";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = 0;
    option.rewardMax = 10;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/a2cnbatch_test15";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACSKCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
//    dqn.save();
}

void test19(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACSKCnnBmNet model(outputNum);
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
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.statPathPrefix = "./a2cnbatch_test19";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test19";
    option.loadModel = false;
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/a2cnbatch_test15";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 5;
    //TODO: testenv
    A2CNStep<AirACSKCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
//    dqn.save();
}

void test20(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACSKCnnBmNet model(outputNum);
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
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.statPathPrefix = "./a2cnbatch_test20";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test20";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/experiments/a2c/docs/smallvalueloss/a2cnbatch_test14";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACSKCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
//    dqn.save();
}

void test21(const int batchSize, const int epochNum) {
	const std::string envName = "LunarLander-v2";
//	const std::string envName = "CartPoleNoFrameskip-v4";
	const int outputNum = 4;
	const int inputNum = 8;

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

    at::IntArrayRef inputShape{num, inputNum};
    DqnOption option(inputShape, deviceType);
    option.gamma = 0.99;
    option.batchSize = batchSize;
    option.targetUpdate = 1024;
    option.entropyCoef = 0.01;
    option.statPathPrefix = "./a2cnbatch_test21";
    option.saveModel = false;
    option.loadModel = false;
    option.inputScale = 1;
    option.isAtari = false;

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    A2CNStep<CartACFcNet, LunarEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
//    dqn.save();
}

void test22(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
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
    		torch::optim::RMSpropOptions(1e-5).eps(1e-5).alpha(0.99));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.statPathPrefix = "./a2cnbatch_test22";
    option.saveModel = false;
    option.savePathPrefix = "./a2cnbatch_test22";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/experiments/a2c/docs/smallvalueloss/a2cnbatch_test14";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
//    dqn.save();
}

void test23(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
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
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.statPathPrefix = "./a2cnbatch_test23";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test23";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/experiments/a2c/docs/smallvalueloss/a2cnbatch_test14";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 5;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, false);
//    dqn.save();
}

void test24(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
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
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.statPathPrefix = "./a2cnbatch_test24";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test24";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/a2cnbatch_test23";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 5;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, false);
//    dqn.save();
}

void test25(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
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
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.statPathPrefix = "./a2cnbatch_test25";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test25";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/a2cnbatch_test24";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 5;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, false);
//    dqn.save();
}

void test26(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
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
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.statPathPrefix = "./a2cnbatch_test26";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test26";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/a2cnbatch_test25";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 5;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, false);
//    dqn.save();
}

void test27(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACSKCnnBmNet model(outputNum);
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
    option.entropyCoef = 0.02;
    option.valueCoef = 0.25;
    option.isAtari = true;
    option.statPathPrefix = "./a2cnbatch_test27";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test27";
    option.loadModel = false;
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/experiments/a2c/docs/smallvalueloss/a2cnbatch_test15";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACSKCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
//    dqn.save();
}

void test28(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACSKCnnBmNet model(outputNum);
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
    option.entropyCoef = 0.005;
    option.valueCoef = 0.25;
    option.isAtari = true;
    option.statPathPrefix = "./a2cnbatch_test28";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test28";
    option.loadModel = false;
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/experiments/a2c/docs/smallvalueloss/a2cnbatch_test15";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACSKCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
//    dqn.save();
}

//L1 loss
//TODO: replace pow(2) by LMSE loss
void test29(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACSKCnnBmNet model(outputNum);
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
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.isAtari = true;
    option.statPathPrefix = "./a2cnbatch_test29";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test29";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/experiments/a2c/docs/smallvalueloss/a2cnbatch_test15";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACSKCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
//    dqn.save();
}

void test30(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACSKCnnBmNet model(outputNum);
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
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.isAtari = true;
    option.statPathPrefix = "./a2cnbatch_test30";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test30";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/a2cnbatch_test29";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACSKCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
//    dqn.save();
}

void test31(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACSKCnnBmNet model(outputNum);
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
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.isAtari = true;
    option.statPathPrefix = "./a2cnbatch_test31";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test31";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/a2cnbatch_test30";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACSKCnnBmNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
//    dqn.save();
}

//Seed updated into 42
void test32(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
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
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.isAtari = true;
    option.statPathPrefix = "./a2cnbatch_test32";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test32";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/a2cnbatch_test30";

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 16;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
//    dqn.save();
}

//update state capacity into 128 instead of 1024
void test33(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.isAtari = true;
    option.statCap = 128;
    option.statPathPrefix = "./a2cnbatch_test33";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test33";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 16;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
//    dqn.save();
}

//update state capacity into 128 instead of 1024
//Adam
//grad norm = 0.1
void test34(const int batchSize, const int epochNum) {
//	const std::string envName = "Breakout-v1";
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5).eps(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.isAtari = true;
    option.statCap = 128;
    option.statPathPrefix = "./a2cnbatch_test34";
    option.saveModel = true;
    option.savePathPrefix = "./a2cnbatch_test34";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.maxGradNormClip = 0.1;
    option.loadModel = false;
    option.loadOptimizer = false;

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 16;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
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
//	test26(atoi(argv[1]), atoi(argv[2]));
//	test27(atoi(argv[1]), atoi(argv[2]));
//	test34(atoi(argv[1]), atoi(argv[2]));
//	testCal(atoi(argv[1]), atoi(argv[2]));

//	testtest14(atoi(argv[1]), atoi(argv[2]));

	LOG4CXX_INFO(logger, "End of test");
}
