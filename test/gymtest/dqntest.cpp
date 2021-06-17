/*
 * dqntest.cpp
 *
 *  Created on: Apr 10, 2021
 *      Author: zf
 */



#include "alg/dqn.hpp"
#include "alg/nbrbdqn.hpp"
#include "alg/dqntarget.hpp"
#include "alg/dqntargetonline.hpp"
#include "alg/dqnsingle.hpp"

#include "gymtest/env/airenv.h"
#include "gymtest/airnets/aircnnnet.h"
#include "gymtest/airnets/aircnnbmnet.h"
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
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("dqntest"));
const torch::Device deviceType = torch::kCUDA;

void test0 (const int clientNum, const int epochNum) {
	const std::string envName = "Alien-v0";
	const int outputNum = 18;

	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();

	AirCnnNet model(outputNum);
	model.to(deviceType);
    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3));
    //	torch::optim::RMSprop optimizer(net.parameters(), torch::optim::RMSpropOptions(rmsLr).eps(1e-8).alpha(0.99));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
    RawPolicy policy(0.1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 128, 0.99);
    option.statPath = "./stat_vanila.txt";


    Dqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adagrad> dqn(model, env, testEnv, policy, optimizer, option);

    dqn.train(clientNum, epochNum);
}

void test1 (const int clientNum, const int epochNum) {
	const std::string envName = "Alien-v0";
	const int outputNum = 18;

	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();

	AirCnnNet model(outputNum);
	model.to(deviceType);
    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3));
    RawPolicy policy(0.1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 128 * clientNum);
    option.statPath = "./stat_singlebatch.txt";


    NbRbDqn<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adagrad> dqn(model, env, testEnv, policy, optimizer, option);

    dqn.train(clientNum, epochNum);
}

void test2(const int clientNum, const int epochNum) {
	const std::string envName = "Alien-v0";
	const int outputNum = 18;

	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
//	testEnv.init();

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3));
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00024).eps(0.01).alpha(0.95));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 128, 0.99);
    option.statPath = "./stat_target.txt";


    DqnTarget<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, testEnv, policy, optimizer, option);

    dqn.train(clientNum, epochNum);
}

void test3(const int clientNum, const int epochNum) {
	const std::string envName = "Alien-v0";
	const int outputNum = 18;

	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 128, 0.99);
    option.targetUpdate = 256;
    option.statPath = "./stat_targetol.txt";


    DqnTargetOnline<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, testEnv, policy, optimizer, option);

    dqn.train(clientNum, epochNum);
}

//102400 means too few exploration
void test4(const int clientNum, const int epochNum) {
	const std::string envName = "BreakoutNoFrameskip-v0";
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

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);
    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 512;
    option.statPath = "./dqn_test4.txt";
    option.teststatPath = "testsingle_stat";
    option.rbCap = 40960;
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test4";
    option.exploreBegin = 1;
    option.exploreEnd = 0.1;
    option.exploreStep = epochNum * 10 / 2;
    option.inputScale = 255;

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adagrad> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(32, epochNum);
}

void test5(const int clientNum, const int epochNum) {
	const std::string envName = "SpaceInvaders-v0";
	const int outputNum = 6;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, num);
	testEnv.init();

	AirCnnBmNet model(outputNum);
	model.to(deviceType);
	AirCnnBmNet targetModel(outputNum);
	targetModel.to(deviceType);
    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 512;
    option.statPath = "./stat_single_space.txt";
    option.teststatPath = "./test_single_space.txt";
    option.rbCap = 10240;


    DqnSingle<AirCnnBmNet, AirEnv, RawPolicy, torch::optim::Adagrad> dqn(model, targetModel, env, testEnv, policy, optimizer, option);

    dqn.train(32, epochNum);
}

void test6(const int clientNum, const int epochNum) {
	const std::string envName = "SpaceInvaders-v0";
	const int outputNum = 6;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, num);
	testEnv.init();

	AirCnnBmNet model(outputNum);
	model.to(deviceType);
	AirCnnBmNet targetModel(outputNum);
	targetModel.to(deviceType);
    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 512;
    option.statPath = "./stat_single_space.txt";
    option.teststatPath = "./test_single_space.txt";
    option.rbCap = 10240;
//    option.saveModel = true;
//    option.savePathPrefix = "./test6";
    option.loadModel = true;
    option.loadPathPrefix = "./test6";
    option.exploreBegin = 0.1;
    option.exploreDecay = 0.1;

    RawPolicy policy(option.exploreBegin, outputNum);


    DqnSingle<AirCnnBmNet, AirEnv, RawPolicy, torch::optim::Adagrad> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.load();
    dqn.train(32, epochNum);

//    dqn.save();
}

void test7(const int clientNum, const int epochNum) {
	const std::string envName = "Alien-v0";
	const int outputNum = 18;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, num);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnBmNet model(outputNum);
	model.to(deviceType);
	AirCnnBmNet targetModel(outputNum);
	targetModel.to(deviceType);
    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 512;
    option.statPath = "./stat_bm_single.txt";
    option.teststatPath = "testsingle_bm_stat";
    option.rbCap = 10240;
    option.saveModel = true;
    option.savePathPrefix = "./test7";
    option.exploreBegin = 1;
    option.exploreDecay = 0.1;
    option.exploreEp = epochNum / 4;
    option.explorePhase =  10;

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnBmNet, AirEnv, RawPolicy, torch::optim::Adagrad> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(32, epochNum);
//    dqn.save();
}

void test8(const int clientNum, const int epochNum) {
	const std::string envName = "Alien-v0";
	const int outputNum = 18;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, num);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);
    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 512;
    option.statPath = "./stat_single.txt";
    option.teststatPath = "testsingle_stat";
    option.rbCap = 10240;
    option.saveModel = true;
    option.savePathPrefix = "./test8";
    option.exploreBegin = 1;
    option.exploreDecay = 0.1;
    option.exploreEp = epochNum / 4;
    option.explorePhase =  10;

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adagrad> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(32, epochNum);
//    dqn.save();
}

void test9(const int clientNum, const int epochNum) {
	const std::string envName = "Alien-v0";
	const int outputNum = 18;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, num);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnBmNet model(outputNum);
	model.to(deviceType);
	AirCnnBmNet targetModel(outputNum);
	targetModel.to(deviceType);
    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 512;
    option.statPath = "./stat_bm_single.txt";
    option.teststatPath = "testsingle_bm_stat";
    option.rbCap = 10240;
    option.saveModel = true;
    option.savePathPrefix = "./test9";
    option.exploreBegin = 1;
    option.exploreDecay = 0.1;
    option.exploreEp = epochNum / 4;
    option.explorePhase =  10;
    option.loadModel = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0418/test7";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnBmNet, AirEnv, RawPolicy, torch::optim::Adagrad> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(32, epochNum);
//    dqn.save();
}

//Test large rl for trained model
//TODO: Test large rl without decay exploration
void test10(const int clientNum, const int epochNum) {
	const std::string envName = "Alien-v0";
	const int outputNum = 18;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, num);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnBmNet model(outputNum);
	model.to(deviceType);
	AirCnnBmNet targetModel(outputNum);
	targetModel.to(deviceType);
    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 512;
    option.statPath = "./stat_bm_single.txt";
    option.teststatPath = "testsingle_bm_stat";
    option.rbCap = 10240;
    option.saveModel = true;
    option.savePathPrefix = "./test10";
    option.exploreBegin = 1;
    option.exploreDecay = 0.1;
    option.exploreEp = epochNum / 64;
    option.explorePhase =  10;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0418/test7";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnBmNet, AirEnv, RawPolicy, torch::optim::Adagrad> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(32, epochNum);
//    dqn.save();
}

//Large rl without explore decay
void test11(const int clientNum, const int epochNum) {
	const std::string envName = "Alien-v0";
	const int outputNum = 18;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, num);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnBmNet model(outputNum);
	model.to(deviceType);
	AirCnnBmNet targetModel(outputNum);
	targetModel.to(deviceType);
    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 512;
    option.statPath = "./stat_bm_single.txt";
    option.teststatPath = "testsingle_bm_stat";
    option.rbCap = 10240;
    option.saveModel = true;
    option.savePathPrefix = "./test10";
    option.exploreBegin = 1;
    option.exploreDecay = 0.1;
    option.exploreEp = epochNum / 128;
    option.explorePhase =  10;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0418/test7";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnBmNet, AirEnv, RawPolicy, torch::optim::Adagrad> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(32, epochNum);
//    dqn.save();
}

void test12(const int clientNum, const int epochNum) {
	const std::string envName = "SpaceInvaders-v0";
	const int outputNum = 6;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, num);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnBmNet model(outputNum);
	model.to(deviceType);
	AirCnnBmNet targetModel(outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(0.00025).eps(1e-6).weight_decay(0.99).momentum(0));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 2500;
    option.statPath = "./stat_dqn_test12.txt";
    option.teststatPath = "test_dqn_test12";
    option.rbCap = 10240;
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test12";
    option.exploreBegin = 1;
    option.exploreDecay = 0.1;
    option.exploreEp = epochNum / 2;
    option.explorePhase =  10;
    option.loadModel = false;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0418/test7";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnBmNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(32, epochNum);
//    dqn.save();
}

//Replace	torch::nn::utils::clip_grad_value_(optimizer.parameters(), 1);
void test13(const int clientNum, const int epochNum) {
	const std::string envName = "SpaceInvaders-v0";
	const int outputNum = 6;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, num);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnBmNet model(outputNum);
	model.to(deviceType);
	AirCnnBmNet targetModel(outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(0.00025).eps(1e-6).weight_decay(0.99).momentum(0));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 2048;
    option.statPath = "./stat_dqn_test13.txt";
    option.teststatPath = "test_dqn_test13";
    option.rbCap = 10240;
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test13";
    option.exploreBegin = 1;
    option.exploreDecay = 0.1;
    option.exploreEp = epochNum / 2;
    option.explorePhase =  10;
    option.loadModel = false;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0418/test7";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnBmNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(32, epochNum);
//    dqn.save();
}

void test14(const int batchSize, const int epochNum) {
	const std::string envName = "Breakout-v0";
	const int outputNum = 4;

	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnBmNet model(outputNum);
	model.to(deviceType);
	AirCnnBmNet targetModel(outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(0.00025).eps(1e-6).weight_decay(0.99).momentum(0));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 2048;
    option.statPath = "./stat_dqn_test14.txt";
    option.teststatPath = "test_dqn_test14";
    option.rbCap = 10240;
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test14";
    option.exploreBegin = 1;
    option.exploreDecay = 0.1;
    option.exploreEp = epochNum / 2;
    option.explorePhase =  10;
    option.loadModel = false;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0418/test7";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnBmNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum);
//    dqn.save();
}

void test15(const int batchSize, const int epochNum) {
	const std::string envName = "Breakout-v0";
	const int outputNum = 4;

	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(0.00025).eps(1e-6).weight_decay(0.99).momentum(0));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 2048;
    option.statPath = "./stat_dqn_test15.txt";
    option.teststatPath = "test_dqn_test15";
    option.rbCap = 10240;
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test15";
    option.exploreBegin = 1;
    option.exploreDecay = 0.1;
    option.exploreEp = epochNum / 2;
    option.explorePhase =  10;
    option.loadModel = false;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0418/test7";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum);
//    dqn.save();
}

void test16(const int batchSize, const int epochNum) {
	const std::string envName = "Breakout-v0";
	const int outputNum = 4;

	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);
    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(0.00025).eps(1e-6).weight_decay(0.99).momentum(0));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 512;
    option.statPath = "./stat_dqn_test16.txt";
    option.teststatPath = "test_dqn_test16";
    option.rbCap = 10240;
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test16";
    option.exploreBegin = 1;
    option.exploreDecay = 0.1;
    option.exploreEp = epochNum / 2;
    option.explorePhase =  10;
    option.loadModel = false;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0418/test7";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnNet, AirEnv, RawPolicy, torch::optim::Adagrad> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum);
//    dqn.save();
}

void test17(const int batchSize, const int epochNum) {
	const std::string envName = "Breakout-v0";
	const int outputNum = 4;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(0.00025).eps(1e-2).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 1024;
    option.statPath = "./stat_dqn_test17.txt";
    option.teststatPath = "test_dqn_test17";
    option.rbCap = 10240;
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test17";
    option.exploreBegin = 1;
    option.exploreEnd = 0.1;
    option.exploreDecay = 0.1;
    option.exploreStep = 1000000;
//    option.exploreEp = epochNum / 2;
//    option.explorePhase =  10;
    option.loadModel = false;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0418/test7";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum);
//    dqn.save();
}

void test18(const int clientNum, const int epochNum) {
	const std::string envName = "Alien-v0";
	const int outputNum = 18;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, num);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnBmNet model(outputNum);
	model.to(deviceType);
	AirCnnBmNet targetModel(outputNum);
	targetModel.to(deviceType);
    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 512;
    option.statPath = "./stat_bm_dqn_18.txt";
    option.teststatPath = "./test_bm_dqn_18";
    option.rbCap = 10240;
    option.saveModel = true;
    option.savePathPrefix = "./test18";
    option.exploreBegin = 0.1;
    option.exploreEnd = 0.05;
    option.exploreStep = epochNum * 70 / 2;
//    option.exploreEp = epochNum / 4;
//    option.explorePhase =  10;
    option.loadModel = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0418/test7";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnBmNet, AirEnv, RawPolicy, torch::optim::Adagrad> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(32, epochNum);
//    dqn.save();
}

void test19(const int clientNum, const int epochNum) {
	const std::string envName = "Alien-v0";
	const int outputNum = 18;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, num);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnBmNet model(outputNum);
	model.to(deviceType);
	AirCnnBmNet targetModel(outputNum);
	targetModel.to(deviceType);
    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 512;
    option.statPath = "./stat_bm_dqn_19.txt";
    option.teststatPath = "./test_bm_dqn_19";
    option.rbCap = 10240;
    option.saveModel = true;
    option.savePathPrefix = "./test19";
    option.exploreBegin = 0.1;
    option.exploreEnd = 0.05;
    option.exploreStep = epochNum * 70 / 2; //about 70 steps per episode
//    option.exploreEp = epochNum / 4;
//    option.explorePhase =  10;
    option.loadModel = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0422/test18";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnBmNet, AirEnv, RawPolicy, torch::optim::Adagrad> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(32, epochNum);
//    dqn.save();
}


void test20(const int clientNum, const int epochNum) {
	const std::string envName = "Alien-v0";
	const int outputNum = 18;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, num);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnBmNet model(outputNum);
	model.to(deviceType);
	AirCnnBmNet targetModel(outputNum);
	targetModel.to(deviceType);
    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 512;
    option.statPath = "./stat_bm_dqn_20.txt";
    option.teststatPath = "./test_bm_dqn_20";
    option.rbCap = 10240;
    option.saveModel = true;
    option.savePathPrefix = "./test20";
    option.exploreBegin = 0.1;
    option.exploreEnd = 0.05;
    option.exploreStep = epochNum * 70 / 2; //about 70 steps per episode
//    option.exploreEp = epochNum / 4;
//    option.explorePhase =  10;
    option.loadModel = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0423/test19";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnBmNet, AirEnv, RawPolicy, torch::optim::Adagrad> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(32, epochNum);
//    dqn.save();
}

void test21(const int clientNum, const int epochNum) {
	const std::string envName = "Alien-v0";
	const int outputNum = 18;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, num);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnBmNet model(outputNum);
	model.to(deviceType);
	AirCnnBmNet targetModel(outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 1024;
    option.statPath = "./stat_bm_dqn_21.txt";
    option.teststatPath = "./test_bm_dqn_21";
    option.rbCap = 10240;
    option.saveModel = true;
    option.savePathPrefix = "./test21";
    option.exploreBegin = 1;
    option.exploreEnd = 0.1;
    option.exploreStep = epochNum * 70 / 2; //about 70 steps per episode
//    option.exploreEp = epochNum / 4;
//    option.explorePhase =  10;
    option.loadModel = false;

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnBmNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(32, epochNum);
//    dqn.save();
}

void test22(const int batchSize, const int epochNum) {
	const std::string envName = "Breakout-v0";
	const int outputNum = 4;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(0.00025).eps(1e-2).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 2048;
    option.statPath = "./stat_dqn_test22.txt";
    option.teststatPath = "test_dqn_test22";
    option.rbCap = 10240;
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test22";
    option.exploreBegin = 0.8;
    option.exploreEnd = 0.7;
    option.exploreDecay = 0.1;
    option.exploreStep = epochNum * 10 / 2;
//    option.exploreEp = epochNum / 2;
//    option.explorePhase =  10;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0424/9_8/dqn_test22";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum);
//    dqn.save();
}

void test23(const int batchSize, const int epochNum) {
	const std::string envName = "Breakout-v0";
	const int outputNum = 4;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnBmNet model(outputNum);
	model.to(deviceType);
	AirCnnBmNet targetModel(outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(0.00025).eps(1e-2).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 4096;
    option.statPath = "./stat_dqn_test23.txt";
    option.teststatPath = "test_dqn_test23";
    option.rbCap = 10240;
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test23";
    option.exploreBegin = 1;
    option.exploreEnd = 0.9;
    option.exploreDecay = 0.1;
    option.exploreStep = epochNum * 10 / 2;
//    option.exploreEp = epochNum / 2;
//    option.explorePhase =  10;
    option.loadModel = false;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0424/9_8/dqn_test22";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnBmNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum);
//    dqn.save();
}

void test24(const int batchSize, const int epochNum) {
	const std::string envName = "Breakout-v0";
	const int outputNum = 4;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnBmNet model(outputNum);
	model.to(deviceType);
	AirCnnBmNet targetModel(outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(0.00025).eps(1e-2).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 4096;
    option.statPath = "./stat_dqn_test24.txt";
    option.teststatPath = "test_dqn_test24";
    option.rbCap = 20480;
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test24";
    option.exploreBegin = 1;
    option.exploreEnd = 0.9;
    option.exploreDecay = 0.1;
    option.exploreStep = epochNum * 7 / 2;
//    option.exploreEp = epochNum / 2;
//    option.explorePhase =  10;
    option.loadModel = false;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0424/9_8/dqn_test22";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnBmNet, AirEnv, RawPolicy, torch::optim::Adam> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum);
//    dqn.save();
}

void test25(const int batchSize, const int epochNum) {
	const std::string envName = "Breakout-v0";
	const int outputNum = 4;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnBmNet model(outputNum);
	model.to(deviceType);
	AirCnnBmNet targetModel(outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(0.00025).eps(1e-2).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 4096;
    option.statPath = "./stat_dqn_test25.txt";
    option.teststatPath = "test_dqn_test25";
    option.rbCap = 40960;
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test25";
    option.exploreBegin = 1;
    option.exploreEnd = 0.9;
    option.exploreDecay = 0.1;
    option.exploreStep = epochNum * 7 / 2;
//    option.exploreEp = epochNum / 2;
//    option.explorePhase =  10;
    option.loadModel = false;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0424/9_8/dqn_test22";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnBmNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum);
//    dqn.save();
}

void test26(const int batchSize, const int epochNum) {
	const std::string envName = "Breakout-v0";
	const int outputNum = 4;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnBmNet model(outputNum);
	model.to(deviceType);
	AirCnnBmNet targetModel(outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(0.00025).eps(1e-2).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 4096;
    option.statPath = "./stat_dqn_test26.txt";
    option.teststatPath = "test_dqn_test27";
    option.rbCap = 81920;
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test26";
    option.exploreBegin = 0.9;
    option.exploreEnd = 0.8;
    option.exploreDecay = 0.1;
    option.exploreStep = epochNum * 8 / 2;
//    option.exploreEp = epochNum / 2;
//    option.explorePhase =  10;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqn_test25";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnBmNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum);
//    dqn.save();
}

void test27(const int batchSize, const int epochNum) {
	const std::string envName = "Breakout-v0";
	const int outputNum = 4;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnBmNet model(outputNum);
	model.to(deviceType);
	AirCnnBmNet targetModel(outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(0.00025).eps(1e-2).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 4096;
    option.statPath = "./stat_dqn_test27.txt";
    option.teststatPath = "test_dqn_test27";
    option.rbCap = 4096;
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test27";
    option.exploreBegin = 0.8;
    option.exploreEnd = 0.7;
    option.exploreDecay = 0.1;
    option.exploreStep = epochNum * 8 / 2;
//    option.exploreEp = epochNum / 2;
//    option.explorePhase =  10;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqn_test26";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnBmNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum);
//    dqn.save();
}

void test28(const int clientNum, const int epochNum) {
	const std::string envName = "BreakoutNoFrameskip-v0";
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
    option.targetUpdate = 512;
    option.statPathPrefix = "./dqn_test28.txt";
    option.rbCap = 40960;
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test28";
    option.exploreBegin = 0.6;
    option.exploreEnd = 0.1;
    option.exploreStep = epochNum * 25 * 3 / 4;
    option.inputScale = 255;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqn_test4";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(32, epochNum);
}

//batch = 64
//to increase entropy
void test29(const int batchNum, const int epochNum) {
	const std::string envName = "BreakoutNoFrameskip-v0";
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
    option.targetUpdate = 512;
    option.statPathPrefix = "./dqn_test29.txt";
    option.rbCap = 40960;
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test29";
    option.exploreBegin = 0.3;
    option.exploreEnd = 0.2;
    option.exploreStep = epochNum * 20 * 3 / 4;
    option.inputScale = 255;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqn_test28";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(batchNum, epochNum);
}

void test30(const int clientNum, const int epochNum) {
	const std::string envName = "BreakoutNoFrameskip-v0";
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

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 1024;
    option.statPath = "./dqn_test30.txt";
//    option.teststatPath = "testsingle_stat";
    option.rbCap = 40960;
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test30";
    option.exploreBegin = 1;
    option.exploreEnd = 0.5;
    option.exploreStep = epochNum * 12 / 2;
    option.inputScale = 255;

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(32, epochNum);
}

//updated rawpolicy
void test31(const int clientNum, const int epochNum) {
	const std::string envName = "BreakoutNoFrameskip-v0";
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

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99)); //baseline parameters
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 1024;
    option.statPath = "./dqn_test31.txt";
//    option.teststatPath = "testsingle_stat";
    option.rbCap = 40960;
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test31";
    option.exploreBegin = 1;
    option.exploreEnd = 0.5;
    option.exploreStep = epochNum * 12 / 2;
    option.inputScale = 255;

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(32, epochNum);
}

void test32(const int clientNum, const int epochNum) {
	const std::string envName = "BreakoutNoFrameskip-v0";
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

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99)); //baseline parameters
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 1024;
    option.statPath = "./dqn_test32.txt";
//    option.teststatPath = "testsingle_stat";
    option.rbCap = 40960;
    option.saveModel = true;
    option.savePathPrefix = "./dqn_test32";
    option.exploreBegin = 0.5;
    option.exploreEnd = 0.4;
    option.exploreStep = epochNum * 12 / 2;
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/dqn_test31";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, env, policy, optimizer, option);

    dqn.train(32, epochNum);
}

void testCal(const int batchSize, const int epochNum) {
	const std::string envName = "Breakout-v0";
	const int outputNum = 4;
	const int clientNum = 1;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirCnnNet model(outputNum);
	model.to(deviceType);
	AirCnnNet targetModel(outputNum);
	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(0.00025).eps(1e-2).alpha(0.95));
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.targetUpdate = 1536;
    option.statPath = "./stat_dqn_test22.txt";
    option.teststatPath = "test_dqn_test22";
    option.rbCap = 128;
    option.saveModel = false;
    option.savePathPrefix = "./dqn_test22";
    option.exploreBegin = 0.8;
    option.exploreEnd = 0.7;
    option.exploreDecay = 0.1;
    option.exploreStep = epochNum * 10 / 2;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.loadModel = false;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0424/9_8/dqn_test22";

    RawPolicy policy(option.exploreBegin, outputNum);

    DqnSingle<AirCnnNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum);
//    dqn.save();
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
    //	log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getDebug());
    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getInfo());
    log4cxx::LogManager::getLoggerRepository()->setConfigured(true);
}
}

int main(int argc, char** argv) {
	logConfigure(false);

//	test0(atoi(argv[1]), atoi(argv[2]));
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
//	test27(atoi(argv[1]), atoi(argv[2]));
//	testCal(atoi(argv[1]), atoi(argv[2]));
//	test28(atoi(argv[1]), atoi(argv[2]));
//	test29(atoi(argv[1]), atoi(argv[2]));
//	test30(atoi(argv[1]), atoi(argv[2]));
//	test31(atoi(argv[1]), atoi(argv[2]));
	test32(atoi(argv[1]), atoi(argv[2]));



	LOG4CXX_INFO(logger, "End of test");
}
