/*
 * ppopong.cpp
 *
 *  Created on: Jun 25, 2021
 *      Author: zf
 */



#include "alg/pposhared.hpp"
#include "alg/a2cnstepnorm.hpp"
#include "alg/pposharedtest.hpp"
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
#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/simplelayout.h>
#include <log4cxx/logmanager.h>

#include <vector>
#include "alg/utils/dqnoption.h"

namespace {
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("pposharedpong"));
const torch::Device deviceType = torch::kCUDA;

void test0(const int updateNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 8; //8
	const int outputNum = 2;
	const int inputNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartACFcNet model(inputNum, outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = false;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedpong_test0";
    option.saveModel = false;
    option.savePathPrefix = "./pposharedpong_test0";
    option.toTest = false;
    option.inputScale = 1;
    option.batchSize = 4;
    option.envNum = clientNum;
    option.epochNum = 4; //4
    option.trajStepNum = 200; //200 //TODO:
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;

    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
//    PPOSharedTest<CartACFcNet, LunarEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    PPORandom<CartACFcNet, LunarEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//nice curve
void test1(const int updateNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 50;
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
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedpong_test1";
    option.saveModel = false;
    option.savePathPrefix = "./pposharedpong_test1";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = 32;
    option.envNum = clientNum;
    option.epochNum = 10;
    option.trajStepNum = option.batchSize * 4;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;

    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOShared<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//Switching from -21 to 18 to -21
void test2(const int updateNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 50;
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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedpong_test2";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedpong_test2";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = 32;
    option.envNum = clientNum;
    option.epochNum = 10;
    option.trajStepNum = option.batchSize * 4;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;

    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOShared<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//From -21 to 10, then stop improvement
void test3(const int updateNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 50;
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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedpong_test3";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedpong_test3";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = 32;
    option.envNum = clientNum;
    option.epochNum = 20;
    option.trajStepNum = option.batchSize * 4;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;

    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOShared<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//promising
void test4(const int updateNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 50;
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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedpong_test4";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedpong_test4";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = 32;
    option.envNum = clientNum;
    option.epochNum = 20;
    option.trajStepNum = option.batchSize * 4;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedpong_test3";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOShared<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}


void testtest4(const int batchSize, const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
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
    option.statPathPrefix = "./ponga2cnbatch_testtest4";
    option.saveModel = false;
    option.savePathPrefix = "./ponga2cnbatch_testtest4";
    option.toTest = true;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.toTest = true;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedpong_test4";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    PPOShared<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.load();
    ppo.test(batchSize, epochNum);
}

//kl clip
//corruption
void test5(const int updateNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 50;
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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedpong_test5";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedpong_test5";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = 32;
    option.envNum = clientNum;
    option.epochNum = 20;
    option.trajStepNum = option.batchSize * 4;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.klEarlyStop = true;
    option.maxKl = 0.01;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedpong_test4";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOShared<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//value clip
//No improvement
void test6(const int updateNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 50;
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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedpong_test6";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedpong_test6";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = 32;
    option.envNum = clientNum;
    option.epochNum = 20;
    option.trajStepNum = option.batchSize * 4;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.klEarlyStop = true;
    option.maxKl = 0.01;
    option.valueClip = true;
    option.maxValueDelta = 1;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedpong_test4";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOShared<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}


//value clip
//No improvement
void test7(const int updateNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 50;
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
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedpong_test7";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedpong_test7";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = 32;
    option.envNum = clientNum;
    option.epochNum = 20;
    option.trajStepNum = option.batchSize * 4;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.klEarlyStop = true;
    option.maxKl = 0.01;
    option.valueClip = false;
    option.maxValueDelta = 1;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedpong_test6";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOShared<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//catastrophy in the middle
void test8(const int updateNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 50;
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
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedpong_test8";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedpong_test8";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = 32;
    option.envNum = clientNum;
    option.epochNum = 20;
    option.trajStepNum = option.batchSize * 4;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.klEarlyStop = true;
    option.maxKl = 0.01;
    option.valueClip = false;
    option.maxValueDelta = 1;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedpong_test7";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOShared<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

void testtestAC(const int batchSize, const int epochNum) {
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
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
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
    option.statPathPrefix = "./ponga2cnbatch_testtest7";
    option.saveModel = false;
    option.savePathPrefix = "./ponga2cnbatch_testtest7";
    option.toTest = true;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.toTest = true;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedpong_test7";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    PPOShared<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.load();
    ppo.test(batchSize, epochNum);
}

void testtestHO(const int batchSize, const int epochNum) {
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
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
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
    option.statPathPrefix = "./pongpposhared_testtest14";
    option.saveModel = false;
    option.savePathPrefix = "./ponga2cnbatch_testtest14";
    option.toTest = true;
    option.inputScale = 256;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.toTest = true;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedpong_test14";


    SoftmaxPolicy policy(outputNum);
    const int maxStep = 10;
    //TODO: testenv
    PPORandom<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.load();
    ppo.test(batchSize, epochNum);
}
void test100(const int updateNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
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
    option.statPathPrefix = "./pposharedpong_test100";
    option.saveModel = false;
    option.savePathPrefix = "./pposharedpong_test100";
    option.toTest = false;
    option.inputScale = 255;
    option.batchSize = 32;
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
    option.loadModel = false;
    option.loadOptimizer = false;

    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOShared<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//Test gae no normalize
void test9(const int updateNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 50;
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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedpong_test9";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedpong_test9";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = 32;
    option.envNum = clientNum;
    option.epochNum = 20;
    option.trajStepNum = option.batchSize * 4;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;

    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOSharedTest<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}


void test10(const int updateNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 50;
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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedpong_testdetach";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedpong_testdetach";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = 32;
    option.envNum = clientNum;
    option.epochNum = 20;
    option.trajStepNum = option.batchSize * 4;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;

    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPORandom<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

void test11(const int updateNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
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

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedpong_test11";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedpong_test11";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = 10;
    option.envNum = clientNum;
    option.epochNum = 10;
    option.trajStepNum = option.batchSize * 10;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;

    SoftmaxPolicy policy(outputNum);
    PPORandom<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

void test12(const int updateNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
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

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.005;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedpong_test12";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedpong_test12";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = 10;
    option.envNum = clientNum;
    option.epochNum = 10;
    option.trajStepNum = option.batchSize * 10;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;

    SoftmaxPolicy policy(outputNum);
    PPORandom<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

void test13(const int updateNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.0003));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.002;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedpong_test13";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedpong_test13";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = 4;
    option.envNum = clientNum;
    option.epochNum = 4;
    option.trajStepNum = option.batchSize * 32;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;

    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPORandom<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

//Train with non-baseline value estimation
void test14(const int updateNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.0003));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.002;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedpong_test14";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedpong_test14";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = 4;
    option.envNum = clientNum;
    option.epochNum = 4;
    option.trajStepNum = option.batchSize * 32;
    option.tdValue = false;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;

    SoftmaxPolicy policy(outputNum);
    PPORandom<AirACHONet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
}

}

namespace {

void testLog(const int updateNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int clientNum = 50;
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
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = true;
    option.statCap = 128;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.25;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./pposharedpong_testlog";
    option.saveModel = true;
    option.savePathPrefix = "./pposharedpong_testlog";
    option.toTest = false;
    option.inputScale = 256;
    option.batchSize = 32;
    option.envNum = clientNum;
    option.epochNum = 20;
    option.trajStepNum = option.batchSize * 4;
    option.ppoLambda = 0.95;
    option.ppoEpsilon = 0.1;
    option.gamma = 0.99;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/pposharedpong_test3";


    SoftmaxPolicy policy(outputNum);
    //TODO: testenv
    PPOShared<AirACCnnNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(updateNum);
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
//    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getDebug());
    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getInfo());
    log4cxx::LogManager::getLoggerRepository()->setConfigured(true);
}
}

int main(int argc, char** argv) {
	logConfigure(false);

	test10(atoi(argv[1]));
//	test1(atoi(argv[1]));
//	test4(atoi(argv[1]));
//	test3(atoi(argv[1]), atoi(argv[2]));
//	test4(atoi(argv[1]), atoi(argv[2]));
//	test8(atoi(argv[1]));
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
//	test22(atoi(argv[1]), atoi(argv[2]));
//	test23(atoi(argv[1]), atoi(argv[2]));
//	testtest7(atoi(argv[1]), atoi(argv[2]));
//	test26(atoi(argv[1]), atoi(argv[2]));
//	test27(atoi(argv[1]), atoi(argv[2]));
//	testtest20(atoi(argv[1]), atoi(argv[2]));
//	test21(atoi(argv[1]), atoi(argv[2]));
//	test29(atoi(argv[1]), atoi(argv[2]));
//	testtest32(atoi(argv[1]), atoi(argv[2]));
//	test37(atoi(argv[1]), atoi(argv[2]));
//	test36(atoi(argv[1]), atoi(argv[2]));
//	testCal(atoi(argv[1]), atoi(argv[2]));
//	testSave(atoi(argv[1]), atoi(argv[2]));


//	testtestHO(atoi(argv[1]), atoi(argv[2]));

	LOG4CXX_INFO(logger, "End of test");
}
