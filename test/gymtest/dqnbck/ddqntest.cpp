/*
 * ddqntest.cpp
 *
 *  Created on: Apr 19, 2021
 *      Author: zf
 */



#include "alg/doubledqnsingle.hpp"

#include "gymtest/env/airenv.h"
#include "gymtest/airnets/aircnnnet.h"
#include "gymtest/airnets/aircnnbmnet.h"
#include "gymtest/train/rawpolicy.h"
#include "alg/dqnoption.h"

#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <vector>

namespace {
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("ddqntest"));
const torch::Device deviceType = torch::kCUDA;

void test0(const int clientNum, const int epochNum) {
	const std::string envName = "Breakout-v0";
	const int outputNum = 4;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
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
    //    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
    torch::optim::RMSprop optimizer(model.parameters(),
    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.gamma = 0.99;
    option.targetUpdate = 512;
    option.statPath = "./stat_bm_ddqn.txt";
    option.teststatPath = "testsingle_bm_ddqn_stat";
    option.rbCap = 10240;
    option.saveModel = true;
    option.savePathPrefix = "./ddqn_test0";
    option.exploreBegin = 1;
    option.exploreEnd = 0.1;
//    option.exploreDecay = 0.1;
    option.exploreStep = epochNum * 8 / 2;
//    option.exploreBegin = 1;
//    option.exploreDecay = 0.1;
    option.exploreEp = epochNum / 2;
    option.explorePhase =  10;
    option.loadModel = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0418/test7";

    RawPolicy policy(option.exploreBegin, outputNum);

    DDqnSingle<AirCnnBmNet, AirEnv, RawPolicy, torch::optim::RMSprop> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(32, epochNum);
//    dqn.save();
}

void test1(const int batchSize, const int epochNum) {
	const std::string envName = "Alien-v0";
	const int outputNum = 18;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
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
    DqnOption option(inputShape, deviceType);
    option.gamma = 0.99;
    option.targetUpdate = 512;
    option.statPath = "./stat_bm_ddqn_test1.txt";
    option.teststatPath = "testsingle_bm_ddqn_stat_test1";
    option.rbCap = 10240;
    option.saveModel = true;
    option.savePathPrefix = "./ddqn_test1";
    option.exploreBegin = 1;
    option.exploreDecay = 0.1;
    option.exploreEp = epochNum / 256;
    option.explorePhase =  10;
    option.loadModel = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0419/ddqn_test0_8192";

    RawPolicy policy(option.exploreBegin, outputNum);

    DDqnSingle<AirCnnBmNet, AirEnv, RawPolicy, torch::optim::Adagrad> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum);
//    dqn.save();
}

//targetupdate = 2048
void test2(const int batchSize, const int epochNum) {
	const std::string envName = "Alien-v0";
	const int outputNum = 18;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
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
    DqnOption option(inputShape, deviceType);
    option.gamma = 0.99;
    option.targetUpdate = 2048; //dqn paper recommended 10,000 frames, ddqn paper recommended 30,000 frames
    option.statPath = "./stat_bm_ddqn_test2.txt";
    option.teststatPath = "testsingle_bm_ddqn_stat_test2";
    option.rbCap = 10240;
    option.saveModel = true;
    option.savePathPrefix = "./ddqn_test2";
    option.exploreBegin = 1;
    option.exploreDecay = 0.1;
    option.exploreEp = epochNum / 256;
    option.explorePhase =  10;
    option.loadModel = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0420/ddqn_test2";

    RawPolicy policy(option.exploreBegin, outputNum);

    DDqnSingle<AirCnnBmNet, AirEnv, RawPolicy, torch::optim::Adagrad> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum);
//    dqn.save();
}

//Replace	torch::nn::utils::clip_grad_value_(optimizer.parameters(), 1);

//targetupdate = 2048
void test3(const int batchSize, const int epochNum) {
	const std::string envName = "Alien-v0";
	const int outputNum = 18;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
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
    DqnOption option(inputShape, deviceType);
    option.gamma = 0.99;
    option.targetUpdate = 2048; //dqn paper recommended 10,000 frames, ddqn paper recommended 30,000 frames
    option.statPath = "./stat_bm_ddqn_test3.txt";
    option.teststatPath = "testsingle_bm_ddqn_stat_test3";
    option.rbCap = 10240;
    option.saveModel = true;
    option.savePathPrefix = "./ddqn_test3";
    option.exploreBegin = 1;
    option.exploreDecay = 0.1;
    option.exploreEp = epochNum / 2;
    option.explorePhase =  10;
    option.loadModel = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0420/ddqn_test2";

    RawPolicy policy(option.exploreBegin, outputNum);

    DDqnSingle<AirCnnBmNet, AirEnv, RawPolicy, torch::optim::Adagrad> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(batchSize, epochNum);
//    dqn.save();
}

void test4(const int clientNum, const int epochNum) {
	const std::string envName = "Alien-v0";
	const int outputNum = 18;

	const int num = 1;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, num);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
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
    DqnOption option(inputShape, deviceType);
    option.gamma = 0.99;
    option.targetUpdate = 512;
    option.statPath = "./stat_bm_ddqn_test4.txt";
    option.teststatPath = "test_bm_ddqn_test4";
    option.rbCap = 10240;
    option.saveModel = true;
    option.savePathPrefix = "./ddqn_test3";
    option.exploreBegin = 1;
    option.exploreDecay = 0.1;
    option.exploreEp = epochNum / 2;
    option.explorePhase =  10;
    option.loadModel = false;
//    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/experiments/0418/test7";

    RawPolicy policy(option.exploreBegin, outputNum);

    DDqnSingle<AirCnnBmNet, AirEnv, RawPolicy, torch::optim::Adagrad> dqn(model, targetModel, env, testEnv, policy, optimizer, option);
    dqn.train(32, epochNum);
//    dqn.save();
}
}
//TODO: Try RMS optimizer with epsilon = 0.95

int main(int argc, char** argv) {
	log4cxx::BasicConfigurator::configure();
//	log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getDebug());
	log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getInfo());

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

	LOG4CXX_INFO(logger, "End of test");
}
