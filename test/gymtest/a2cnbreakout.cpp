/*
 * a2cnbreakout.cpp
 *
 *  Created on: Jun 18, 2021
 *      Author: zf
 */




#include "alg/a2cnstep.hpp"
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
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("a2cnbatchpong"));
const torch::Device deviceType = torch::kCUDA;


void test0(const int epochNum) {
	const int batchSize = 50;

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
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
    option.statPathPrefix = "./boa2cnbatch_test0";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test0";
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
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}

void testtest0(const int batchSize, const int epochNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
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
    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.1;
    option.statPathPrefix = "./breakouta2cnbatch_testtest0";
    option.saveModel = false;
    option.savePathPrefix = "./breakouta2cnbatch_testtest0";
    option.toTest = true;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test0";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.load();
    a2c.test(batchSize, epochNum);
}

void test1(const int epochNum) {
	const int batchSize = 50;

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
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.1;
    option.statPathPrefix = "./boa2cnbatch_test1";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test1";
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
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}

void testtest1(const int epochNum) {
	const int batchSize = 16;

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
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_testtest1";
    option.saveModel = false;
    option.savePathPrefix = "./boa2cnbatch_testtest1";
    option.toTest = true;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test1";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.load();
    a2c.test(batchSize, epochNum);
//    a2c.train(epochNum, true);
}


//Not converge
void test2(const int epochNum) {
	const int batchSize = 50;

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
    option.statCap = 128;
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
    option.statPathPrefix = "./boa2cnbatch_test2";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test2";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1000;
    option.rewardMax = 1000;
    option.normReward = true;
    option.loadModel = false;
    option.loadOptimizer = false;

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 16;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}

//No better than test0
//normReward = true;
void test3(const int epochNum) {
	const int batchSize = 50;

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
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
    option.statPathPrefix = "./boa2cnbatch_test3";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test3";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test0";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 16;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}

//No better than test0. maybe too small epochNum
void test4(const int epochNum) {
	const int batchSize = 50;

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
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
    option.statPathPrefix = "./boa2cnbatch_test4";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test4";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test0";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 8;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}

//normalized reward
//Unstable
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

	AirACCnnNet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
    option.statPathPrefix = "./boa2cnbatch_test5";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test5";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1000;
    option.rewardMax = 1000;
    option.normReward = true;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test0";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 20;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}

//5 lives per episode
//no reward clip
//Wrong
void test6(const int epochNum) {
	const int batchSize = 50;

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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
    option.statPathPrefix = "./boa2cnbatch_test6";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test6";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1000;
    option.rewardMax = 1000;
    option.normReward = true;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test0";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 20;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}

//5 lives per episode
//with reward clip
void test7(const int epochNum) {
	const int batchSize = 50;

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
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
    option.statPathPrefix = "./boa2cnbatch_test7";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test7";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.normReward = false;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test0";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 20;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}

//Not load optimizer, and learn with smaller rl. To detect current lr
//count life end as end of episode
//maxStep = 50
//<= 0.35
void test8(const int epochNum) {
	const int batchSize = 50;

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
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
    option.statPathPrefix = "./boa2cnbatch_test8";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test8";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.normReward = false;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test0";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 40;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}

//Deal with reward
//promissing
void test9(const int epochNum) {
	const int batchSize = 50;

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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_test9";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test9";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test0";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}

//small improvement of test9
void test10(const int epochNum) {
	const int batchSize = 50;

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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_test10";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test10";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test9";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}

//small improvement of test10, but unstable
void test11(const int epochNum) {
	const int batchSize = 50;

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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_test11";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test11";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test10";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}

//Seemed work, don't know purpose
void test12(const int epochNum) {
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
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_test12";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test12";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test10";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}


//better than test11
void test13(const int epochNum) {
	const int batchSize = 50;

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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_test13";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test13";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test11";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}

//OK, but not stable
void test14(const int epochNum) {
	const int batchSize = 50;

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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_test14";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test14";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test13";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}

//OK. unstable
void test15(const int epochNum) {
	const int batchSize = 50;

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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_test15";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test15";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test14";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}


void testtest15(const int epochNum) {
	const int batchSize = 16;

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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_testtest15";
    option.saveModel = false;
    option.savePathPrefix = "./boa2cnbatch_testtest15";
    option.toTest = true;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test15";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.load();
    a2c.test(batchSize, epochNum);
//    a2c.train(epochNum, true);
}

//lr, maxStep, valueCoef
//Bad performance of GAE
//Seemed not converge
void test16(const int epochNum) {
	const int batchSize = 50;

	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10210";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(7e-5).eps(1e-5).alpha(0.99));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.1;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_test16";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test16";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test14";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 20;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}


//OK
void test17(const int epochNum) {
	const int batchSize = 50;

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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_test17";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test17";
    option.saveThreshold = 40;
    option.saveStep = 5;
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test15";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}



void testtest17(const int epochNum) {
	const int batchSize = 16;

	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10210";
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
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_testtest17";
    option.saveModel = false;
    option.savePathPrefix = "./boa2cnbatch_testtest17";
    option.toTest = true;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test17_50.220028";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.load();
    a2c.test(batchSize, epochNum);
//    a2c.train(epochNum, true);
}


//increase batchSize
void test18(const int epochNum) {
	const int batchSize = 80;

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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_test18";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test18";
    option.saveThreshold = 45;
    option.saveStep = 5;
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test17";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}


void testtest18_65(const int epochNum) {
	const int batchSize = 16;

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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_testtest18_65";
    option.saveModel = false;
    option.savePathPrefix = "./boa2cnbatch_testtest18_65";
    option.toTest = true;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test18_65.106262";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.load();
    a2c.test(batchSize, epochNum);
//    a2c.train(epochNum, true);
}

void testtest18_60(const int epochNum) {
	const int batchSize = 16;

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
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_testtest18_60";
    option.saveModel = false;
    option.savePathPrefix = "./boa2cnbatch_testtest18_60";
    option.toTest = true;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test18_60.118771";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.load();
    a2c.test(batchSize, epochNum);
//    a2c.train(epochNum, true);
}

void testtest18_55(const int epochNum) {
	const int batchSize = 16;

	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10205";
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
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_testtest18_55";
    option.saveModel = false;
    option.savePathPrefix = "./boa2cnbatch_testtest18_55";
    option.toTest = true;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test18_55.025028";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.load();
    a2c.test(batchSize, epochNum);
//    a2c.train(epochNum, true);
}


void testtest18_50(const int epochNum) {
	const int batchSize = 16;

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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_testtest18_50";
    option.saveModel = false;
    option.savePathPrefix = "./boa2cnbatch_testtest18_50";
    option.toTest = true;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test18_50.225002";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.load();
    a2c.test(batchSize, epochNum);
//    a2c.train(epochNum, true);
}



void testtest18_45(const int epochNum) {
	const int batchSize = 16;

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
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_testtest18_45";
    option.saveModel = false;
    option.savePathPrefix = "./boa2cnbatch_testtest18_45";
    option.toTest = true;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test18_45.012516";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.load();
    a2c.test(batchSize, epochNum);
//    a2c.train(epochNum, true);
}
//TODO: Continue training test18


//Modified lr
void test200(const int epochNum) {
	const int batchSize = 50;

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
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(7e-5).eps(1e-5).alpha(0.99));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.donePerEp = 5;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
    option.statPathPrefix = "./boa2cnbatch_test200";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test3";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test3";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 16;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}

//TBC
void test201(const int epochNum) {
	const int batchSize = 50;

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
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(7e-5).eps(1e-5).alpha(0.99));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.donePerEp = 5;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
    option.statPathPrefix = "./boa2cnbatch_test201";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test201";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test3";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 16;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}

//Continue 201
void test202(const int epochNum) {
	const int batchSize = 50;

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
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(7e-5).eps(1e-5).alpha(0.99));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.donePerEp = 5;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.1;
    option.statPathPrefix = "./boa2cnbatch_test202";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test202";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test201";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 16;
    //TODO: testenv
    A2CNStep<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}

//TODO: Train from start with more steps, clipped reward
void test203(const int epochNum) {
	const int batchSize = 50;

	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10210";
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
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_test203";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test203";
    option.saveThreshold = 40;
    option.saveStep = 5;
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.normReward = false;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test15";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 20;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}

void test204(const int epochNum) {
	const int batchSize = 50;

	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10210";
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
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_test204";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test204";
    option.saveThreshold = 40;
    option.saveStep = 5;
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.normReward = false;
    option.clipRewardStat = true;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test203";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 20;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}


void test205(const int epochNum) {
	const int batchSize = 50;

	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10210";
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
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_test205";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test205";
    option.saveThreshold = 40;
    option.saveStep = 5;
    option.sumSaveThreshold = 250;
    option.sumSaveStep = 10;
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.normReward = false;
    option.clipRewardStat = true;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test204";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 20;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}

void testtest204_95(const int epochNum) {
	const int batchSize = 16;

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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_testtest204_95";
    option.saveModel = false;
    option.savePathPrefix = "./boa2cnbatch_testtest204_95";
    option.toTest = true;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test204_95.400078";

    option.hangRewardTh = 0.5;
    option.randomHang = false;
    option.testOutput = outputNum;
    option.hangNumTh = 500;
    option.randomStep = 5;


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 20;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.load();
    a2c.test(batchSize, epochNum);
//    a2c.train(epochNum, true);
}

void testtest204_90(const int epochNum) {
	const int batchSize = 16;

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
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_testtest204_90";
    option.saveModel = false;
    option.savePathPrefix = "./boa2cnbatch_testtest204_90";
    option.toTest = true;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test204_90.920052";

    option.hangRewardTh = 0.5;
    option.randomHang = true;
    option.testOutput = outputNum;
    option.hangNumTh = 500;
    option.randomStep = 5;

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 20;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.load();
    a2c.test(batchSize, epochNum);
//    a2c.train(epochNum, true);
}


void testtest204_85(const int epochNum) {
	const int batchSize = 16;

	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10205";
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
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_testtest204_85";
    option.saveModel = false;
    option.savePathPrefix = "./boa2cnbatch_testtest204_85";
    option.toTest = true;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test204_86.000046";

    option.hangRewardTh = 0.5;
    option.randomHang = false;
    option.testOutput = outputNum;
    option.hangNumTh = 500;
    option.randomStep = 5;

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 20;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.load();
    a2c.test(batchSize, epochNum);
//    a2c.train(epochNum, true);
}

void testtest204_80(const int epochNum) {
	const int batchSize = 16;

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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_testtest204_80";
    option.saveModel = false;
    option.savePathPrefix = "./boa2cnbatch_testtest204_80";
    option.toTest = true;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test204_80.260094";

    option.hangRewardTh = 0.5;
    option.randomHang = true;
    option.testOutput = outputNum;
    option.hangNumTh = 500;
    option.randomStep = 5;


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 20;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.load();
    a2c.test(batchSize, epochNum);
//    a2c.train(epochNum, true);
}

void testtest204_75(const int epochNum) {
	const int batchSize = 16;

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
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_testtest204_75";
    option.saveModel = false;
    option.savePathPrefix = "./boa2cnbatch_testtest204_75";
    option.toTest = true;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test204_76.180092";

    option.hangRewardTh = 0.5;
    option.randomHang = true;
    option.testOutput = outputNum;
    option.hangNumTh = 500;
    option.randomStep = 5;

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 20;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.load();
    a2c.test(batchSize, epochNum);
//    a2c.train(epochNum, true);
}


void testtest204_70(const int epochNum) {
	const int batchSize = 16;

	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10205";
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
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_testtest204_70";
    option.saveModel = false;
    option.savePathPrefix = "./boa2cnbatch_testtest204_70";
    option.toTest = true;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test204_71.500145";

    option.hangRewardTh = 0.5;
    option.randomHang = true;
    option.testOutput = outputNum;
    option.hangNumTh = 500;
    option.randomStep = 5;

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 20;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.load();
    a2c.test(batchSize, epochNum);
//    a2c.train(epochNum, true);
}

void testtest204(const int epochNum) {
	const int batchSize = 16;

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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_testtest204";
    option.saveModel = false;
    option.savePathPrefix = "./boa2cnbatch_testtest204";
    option.toTest = true;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test204";

    option.hangRewardTh = 0.5;
    option.randomHang = true;
    option.testOutput = outputNum;
    option.hangNumTh = 500;
    option.randomStep = 5;

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 20;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.load();
    a2c.test(batchSize, epochNum);
//    a2c.train(epochNum, true);
}


void testtest205_100(const int epochNum) {
	const int batchSize = 16;

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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-5));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_testtest205_100";
    option.saveModel = false;
    option.savePathPrefix = "./boa2cnbatch_testtest205_100";
    option.toTest = true;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test205_100.730019";

    option.hangRewardTh = 0.5;
    option.randomHang = true;
    option.testOutput = outputNum;
    option.hangNumTh = 500;
    option.randomStep = 5;

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 20;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.load();
    a2c.test(batchSize, epochNum);
//    a2c.train(epochNum, true);
}

void testtest205_120(const int epochNum) {
	const int batchSize = 16;

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
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_testtest205_120";
    option.saveModel = false;
    option.savePathPrefix = "./boa2cnbatch_testtest205_120";
    option.toTest = true;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test205_120.050095";

    option.hangRewardTh = 0.5;
    option.randomHang = true;
    option.testOutput = outputNum;
    option.hangNumTh = 500;
    option.randomStep = 5;

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 20;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.load();
    a2c.test(batchSize, epochNum);
//    a2c.train(epochNum, true);
}

void testtest205_420(const int epochNum) {
	const int batchSize = 16;

	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10205";
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
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_testtest205_420";
    option.saveModel = false;
    option.savePathPrefix = "./boa2cnbatch_testtest205_420";
    option.toTest = true;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test205_421.400177";

    option.hangRewardTh = 0.5;
    option.randomHang = true;
    option.testOutput = outputNum;
    option.hangNumTh = 500;
    option.randomStep = 5;

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 20;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.load();
    a2c.test(batchSize, epochNum);
//    a2c.train(epochNum, true);
}

//idle log
void test206(const int epochNum) {
	const int batchSize = 50;

	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10210";
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
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_test206";
    option.saveModel = true;
    option.savePathPrefix = "./boa2cnbatch_test206";
    option.saveThreshold = 80;
    option.saveStep = 5;
    option.sumSaveThreshold = 2500;
    option.sumSaveStep = 10;
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;

    option.toPunish = true;
    option.penalStep = 100;
    option.penalReward = -0.5;
    option.multiLifes = true;
    option.donePerEp = 5;

    option.normReward = false;
    option.clipRewardStat = true;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test205_421.400177";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 20;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum, true);
}

void testtest206(const int epochNum) {
	const int batchSize = 16;

	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = batchSize;
	std::string serverAddr = "tcp://127.0.0.1:10210";
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
    option.donePerEp = 5;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./boa2cnbatch_testtest206";
    option.saveModel = false;
    option.savePathPrefix = "./boa2cnbatch_testtest206";
    option.toTest = true;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -100;
    option.rewardMax = 100;
    option.normReward = false;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/boa2cnbatch_test206";

    option.hangRewardTh = 0.5;
    option.randomHang = false;
    option.testOutput = outputNum;
    option.hangNumTh = 500;
    option.randomStep = 5;

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 20;
    //TODO: testenv
    A2CNStep<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.load();
    a2c.test(batchSize, epochNum);
//    a2c.train(epochNum, true);
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

//	test0(atoi(argv[1]));
//	test18(atoi(argv[1]));

//Test
//	testtest204_85(atoi(argv[1]));

//GAE
//	test102(atoi(argv[1]));

//Clipped
	testtest206(atoi(argv[1]));

//	testtest0(atoi(argv[1]), atoi(argv[2]));

	LOG4CXX_INFO(logger, "End of test");
}
