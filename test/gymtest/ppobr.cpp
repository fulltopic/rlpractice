/*
 * ppobr.cpp
 *
 *  Created on: Jul 14, 2021
 *      Author: zf
 */



#include "alg/pposhared.hpp"
#include "alg/pposharedtest.hpp"
#include "alg/a2cnstepnorm.hpp"
#include "alg/pporandom.hpp"

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
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("pposharedbreakout"));
const torch::Device deviceType = torch::kCUDA;

void test1(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
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
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test1";
    option.saveModel = false;
    option.savePathPrefix = "./pposharedbr_test1";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = 32; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 10;
    option.trajStepNum = option.batchSize * 4;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.2;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedpong_test7";

    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOShared<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

void test2(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 32;
    const int roundNum = 2;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test2";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test2";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 10;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.2;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;

    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOSharedTest<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//round = 2
void test3(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 32;
    const int roundNum = 2;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test3";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test3";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 10;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.2;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test2";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOSharedTest<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//round = 4
void test4(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 32;
    const int roundNum = 4;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test4";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test4";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 10;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.2;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test3";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOSharedTest<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//round = 1
void test5(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 32;
    const int roundNum = 1;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test5";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test5";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 10;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.2;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test3";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOSharedTest<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//round = 1
void test6(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 32;
    const int roundNum = 1;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test6";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test6";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 10;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.2;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test5";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOSharedTest<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//round = 4
void test7(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 32;
    const int roundNum = 4;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test7";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test7";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 10;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.2;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test4";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOSharedTest<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}
//TODO: Try less envNum but more trajStep, maybe by pong

//TODO: Early stop for round1
//TODO: print out loss output by sample instead of sequence

//round = 1, early stop
void test8(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 32;
    const int roundNum = 1;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test8";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test8";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 10;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.2;
    option.klEarlyStop = true;
    option.maxKl = 0.5;
    option.valueClip = false;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test5";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOSharedTest<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//round = 1, NO early stop, epochNum = 20
void test9(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 32;
    const int roundNum = 1;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test9";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test9";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 20;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.2;
    option.klEarlyStop = false;
//    option.maxKl = 0.5;
    option.valueClip = false;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test5";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOSharedTest<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//round = 1
void test10(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 32;
    const int roundNum = 1;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test10";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test10";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 10;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.2;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test6";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOSharedTest<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//TODO: print clipped reward to check variance
//round = 1
void test11(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 32;
    const int roundNum = 1;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test11";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test11";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 10;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.2;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test10";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOSharedTest<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//round = 1
//Orig template, not pposharedtest, valueCoef = 0.25
void test12(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 32;
    const int roundNum = 1;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test12";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test12";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 10;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test10";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOShared<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//round = 1
//Orig template, not pposharedtest, valueCoef = 0.25
void test13(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 32;
    const int roundNum = 2;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test13";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test13";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 10;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test4";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOShared<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//Orig template, not pposharedtest, valueCoef = 0.5
void test14(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 8;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 32;
    const int roundNum = 4;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test14";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test14";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 4;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test4";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOShared<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//try lucky
//Orig template, not pposharedtest, valueCoef = 0.5
void test15(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 40;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 32;
    const int roundNum = 4;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test15";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test15";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 10;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.2;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test4";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOShared<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//TODO: less lr
void test16(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 40;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 32;
    const int roundNum = 4;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test16";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test16";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 10;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.2;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test4";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOSharedTest<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//TODO: less lr
void test17(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 40;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 32;
    const int roundNum = 4;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test17";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test17";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 10;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.2;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test16";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOSharedTest<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//more epoch
void test18(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 40;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 32;
    const int roundNum = 4;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test18";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test18";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 20;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.2;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test16";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOSharedTest<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}


void test19(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 40;
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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 16;
    const int roundNum = 1;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test19";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test19";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 40;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.2;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test16";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOSharedTest<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

void test20(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 40;
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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 16;
    const int roundNum = 1;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test20";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test20";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 40;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.2;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test19";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOSharedTest<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}


void test21(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 20;
    const int roundNum = 4;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test21";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test21";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 10;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.8;
    option.ppoEpsilon = 0.1;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.normReward = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test10";


    SoftmaxPolicy policy(outputNum);
    PPOSharedTest<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

void test22(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 20;
    const int roundNum = 4;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test22";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test22";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 10;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.8;
    option.ppoEpsilon = 0.1;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.normReward = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test10";


    SoftmaxPolicy policy(outputNum);
    PPOSharedTest<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

void test23(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(2.5e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 20; //from stat, seemed a reward feedback at first 25 steps
    const int roundNum = 1;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test23";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test23";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 1;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.normReward = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test10";


    SoftmaxPolicy policy(outputNum);
    PPOSharedTest<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}


void test24(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(2.5e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 20; //from stat, seemed a reward feedback at first 25 steps
    const int roundNum = 1;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test24";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test24";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 1;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.normReward = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test10";


    SoftmaxPolicy policy(outputNum);
    PPOSharedTest<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//try epoch = 2, compare to test24
void test25(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(2.5e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 20; //from stat, seemed a reward feedback at first 25 steps
    const int roundNum = 1;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test25";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test25";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 2;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.normReward = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test10";


    SoftmaxPolicy policy(outputNum);
    PPOSharedTest<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//round = 2, compare to test24
void test26(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(2.5e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 20; //from stat, seemed a reward feedback at first 25 steps
    const int roundNum = 2;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test26";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test26";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 1;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.normReward = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test10";


    SoftmaxPolicy policy(outputNum);
    PPOSharedTest<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}


void test27(const int updateNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 2;
	const int clientNum = 4;
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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(2.5e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 8; //from stat, seemed a reward feedback at first 25 steps
    const int roundNum = 2;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test27";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test27";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 2;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.normReward = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test10";


    SoftmaxPolicy policy(outputNum);
    PPORandom<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}


void test28(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(2.5e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 20; //from stat, seemed a reward feedback at first 25 steps
    const int roundNum = 1;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test28";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test28";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 1;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.normReward = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test10";


    SoftmaxPolicy policy(outputNum);
    PPORandom<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}


void test29(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(2.5e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 20; //from stat, seemed a reward feedback at first 25 steps
    const int roundNum = 1;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test29";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test29";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 1;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.normReward = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test28";


    SoftmaxPolicy policy(outputNum);
    PPORandom<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

void test30(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(2.5e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 20; //from stat, seemed a reward feedback at first 25 steps
    const int roundNum = 1;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test30";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test30";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 2;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.normReward = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test28";


    SoftmaxPolicy policy(outputNum);
    PPORandom<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}


void test31(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
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
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(2.5e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 20; //from stat, seemed a reward feedback at first 25 steps
    const int roundNum = 2;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test31";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test31";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 1;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.normReward = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test28";


    SoftmaxPolicy policy(outputNum);
    PPORandom<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::RMSprop> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

void test32(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 10; //from stat, seemed a reward feedback at first 25 steps
    const int roundNum = 4;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test32";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test32";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 10;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.normReward = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test28";


    SoftmaxPolicy policy(outputNum);
    PPORandom<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

void test33(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 50;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 10; //from stat, seemed a reward feedback at first 25 steps
    const int roundNum = 10;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_test33";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_test33";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 10;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.normReward = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test32";


    SoftmaxPolicy policy(outputNum);
    PPORandom<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}
//TODO: normalize gae, valueCoef = 0.25


//TODO: valueCoef = 0.5

void testLog(const int updateNum) {
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int clientNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHONet model(outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 4;
    const int roundNum = 2;

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedbr_testLog";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedbr_testLog";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = maxStep; //batchSize = maxStep
    option.envNum = clientNum;
    option.epochNum = 2;
    option.trajStepNum = maxStep * roundNum;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.klEarlyStop = false;
    option.valueClip = false;
    option.normReward = false;
    option.gamma = 0.99;
    option.clipRewardStat = true;
    option.rewardScale = 1;
    option.rewardMin = -10;
    option.rewardMax = 10;
    option.loadModel = false;
    option.loadOptimizer = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedbr_test10";


    SoftmaxPolicy policy(outputNum);
    PPORandom<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
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

	test33(atoi(argv[1]));
//	testLog(atoi(argv[1]));
}
