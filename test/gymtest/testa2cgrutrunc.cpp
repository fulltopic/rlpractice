/*
 * testa2cgrutrunc.cpp
 *
 *  Created on: Feb 8, 2022
 *      Author: zf
 */

#include "alg/rnn/a2cgrutrunc.hpp"
#include "alg/rnn/a2cgrutruncgae.hpp"
#include "alg/rnn/a2cgrutruncslim.hpp"
#include "alg/rnn/a2cgrutruncslimgae.hpp"
#include "alg/utils/dqnoption.h"

#include "gymtest/env/airenv.h"
#include "gymtest/env/lunarenv.h"
#include "gymtest/rnnnets/lunarnets/cartacgrutruncnet.h"
#include "gymtest/rnnnets/lunarnets/cartacgruslim.h"
#include "gymtest/rnnnets/airnets/airacgrunet.h"
#include "gymtest/rnnnets/airnets/airacgruslimnet.h"

#include "gymtest/train/rawpolicy.h"
#include "gymtest/train/softmaxpolicy.h"

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

void testCart(const int epochNum) {
	const int batchSize = 40;
	const std::string envName = "CartPole-v0";
	const int outputNum = 2;
	const int inputNum = 4;
	const int testClientNum = 4;
//	const int maxStep = 8;
	const int hiddenLayerNum = 1;
	const int hiddenNum = 256;

	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, batchSize);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	LunarEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartACGRUTruncFcNet model(inputNum, hiddenNum, outputNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
    LOG4CXX_INFO(logger, "Model ready");


    at::IntArrayRef inputShape{4};
    at::IntArrayRef testInputShape {4};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = batchSize;
    option.isAtari = false;
//    option.envStep = maxStep;
    option.donePerEp = 1;
    option.multiLifes = false;
    //grad
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.gamma = 0.9;
    option.maxStep = 30;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/a2crnn_testtrunccart_m20/tfevents.pb";
    //input
    option.inputScale = 1;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    //test
    option.toTest = true;
    option.testGapEp = 400;
    option.testBatch = testClientNum;
    option.testEp = testClientNum;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./a2cngae_test0";
    //rnn
    option.hiddenNums = {hiddenNum};
    option.hidenLayerNums = {1};
    option.maxStep = 20;
    option.gruCellNum = 1;


    SoftmaxPolicy policy(outputNum);
    A2CGRUTrunc<CartACGRUTruncFcNet, LunarEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, testEnv, policy, optimizer, option);
    a2c.train(epochNum);
}

void testCartSlim(const int epochNum) {
	const int batchSize = 7;
	const std::string envName = "CartPole-v0";
	const int outputNum = 2;
	const int inputNum = 4;
	const int testClientNum = 4;
//	const int maxStep = 8;
	const int hiddenLayerNum = 1;
	const int hiddenNum = 256;

	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, batchSize);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10206";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	LunarEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartACGRUTruncFcSlimNet model(inputNum, hiddenNum, outputNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
    LOG4CXX_INFO(logger, "Model ready");


    at::IntArrayRef inputShape{4};
    at::IntArrayRef testInputShape {4};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = batchSize;
    option.isAtari = false;
//    option.envStep = maxStep;
    option.donePerEp = 1;
    option.multiLifes = false;
    //grad
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.gamma = 0.9;
    option.maxStep = 30;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/a2crnn_testtrunccartslim/tfevents.pb";
    //input
    option.inputScale = 1;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    //test
    option.toTest = true;
    option.testGapEp = 400;
    option.testBatch = testClientNum;
    option.testEp = testClientNum;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./a2cngae_test0";
    //rnn
    option.hiddenNums = {hiddenNum};
    option.hidenLayerNums = {1};
    option.maxStep = 10;
    option.gruCellNum = 1;


    SoftmaxPolicy policy(outputNum);
    A2CGRUTruncSlim<CartACGRUTruncFcSlimNet, LunarEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, testEnv, policy, optimizer, option);
    a2c.train(epochNum);
}


void testCartGae(const int epochNum) {
	const int batchSize = 40;
	const std::string envName = "CartPole-v0";
	const int outputNum = 2;
	const int inputNum = 4;
	const int testClientNum = 4;
//	const int maxStep = 8;
	const int hiddenLayerNum = 1;
	const int hiddenNum = 256;

	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, batchSize);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	LunarEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartACGRUTruncFcNet model(inputNum, hiddenNum, outputNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
    LOG4CXX_INFO(logger, "Model ready");


    at::IntArrayRef inputShape{4};
    at::IntArrayRef testInputShape {4};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = batchSize;
    option.isAtari = false;
//    option.envStep = maxStep;
    option.donePerEp = 1;
    option.multiLifes = false;
    //grad
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.gamma = 0.9;
    option.maxStep = 30;
    //log
    option.logInterval = 100;
    option.tensorboardLogPath = "./logs/a2crnn_testtrunccartgae_m20/tfevents.pb";
    //input
    option.inputScale = 1;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    //test
    option.toTest = true;
    option.testGapEp = 400;
    option.testBatch = testClientNum;
    option.testEp = testClientNum;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./a2cngae_test0";
    //rnn
    option.hiddenNums = {hiddenNum};
    option.hidenLayerNums = {1};
    option.maxStep = 20;
    option.gruCellNum = 1;
    option.ppoLambda = 0.6;


    SoftmaxPolicy policy(outputNum);
    A2CGRUTruncGAE<CartACGRUTruncFcNet, LunarEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, testEnv, policy, optimizer, option);
    a2c.train(epochNum);
}

void testPong(const int epochNum) {
	const int batchSize = 40;
	const int envNum = 37;
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int inputNum = 4;
	const int testClientNum = 8;
//	const int maxStep = 8; //deprecated
//	const int hiddenLayerNum = 1;
//	const int hiddenNum = 256;

	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, envNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHOGRUNet model(outputNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
    LOG4CXX_INFO(logger, "Model ready");


    at::IntArrayRef inputShape{4, 84, 84};
    at::IntArrayRef testInputShape {4, 84, 84};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = envNum;
    option.isAtari = true;
    option.envStep = 8; //deprecated
    option.donePerEp = 1;
    option.multiLifes = false;
    //grad
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.gamma = 0.99;
    option.maxStep = 30;
    //log
    option.logInterval = 10;
    option.tensorboardLogPath = "./logs/a2crnn_testtruncpong_log1/tfevents.pb";
    //input
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    //test
    option.toTest = true;
    option.testGapEp = 100;
    option.testBatch = testClientNum;
    option.testEp = testClientNum;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./a2cngae_test0";
    //rnn
    option.hiddenNums = {3136};
    option.hidenLayerNums = {1};
    option.maxStep = 10;
    option.gruCellNum = 1;


    SoftmaxPolicy policy(outputNum);
    A2CGRUTrunc<AirACHOGRUNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, testEnv, policy, optimizer, option);
    a2c.train(epochNum);
}

void testPongSlim(const int epochNum) {
	const int batchSize = 40;
	const int envNum = 37;
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int inputNum = 4;
	const int testClientNum = 4;
//	const int maxStep = 8; //deprecated
//	const int hiddenLayerNum = 1;
//	const int hiddenNum = 256;

	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, envNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10206";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHOGRUSlimNet model(outputNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
    LOG4CXX_INFO(logger, "Model ready");


    at::IntArrayRef inputShape{4, 84, 84};
    at::IntArrayRef testInputShape {4, 84, 84};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = envNum;
    option.isAtari = true;
    option.envStep = 8; //deprecated
    option.donePerEp = 1;
    option.multiLifes = false;
    //grad
    option.entropyCoef = 0.01;
    option.valueCoef = 0.3;
    option.maxGradNormClip = 0.5;
    option.gamma = 0.99;
    option.maxStep = 30;
    //log
    option.logInterval = 10;
    option.tensorboardLogPath = "./logs/a2crnn_testpongslim/tfevents.pb";
    //input
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    //test
    option.toTest = true;
    option.testGapEp = 100;
    option.testBatch = testClientNum;
    option.testEp = testClientNum;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./a2cngae_test0";
    //rnn
    option.hiddenNums = {3136};
    option.hidenLayerNums = {1};
    option.maxStep = 10;
    option.gruCellNum = 1;


    SoftmaxPolicy policy(outputNum);
    A2CGRUTruncSlim<AirACHOGRUSlimNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, testEnv, policy, optimizer, option);
    a2c.train(epochNum);
}


void testPongGae(const int epochNum) {
	const int envNum = 37;
	const int batchSize = 40;
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
//	const int inputNum = 4;
	const int testClientNum = 4;
//	const int maxStep = 8; //deprecated
//	const int hiddenLayerNum = 1;
//	const int hiddenNum = 256;

	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, envNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHOGRUNet model(outputNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
    LOG4CXX_INFO(logger, "Model ready");


    at::IntArrayRef inputShape{4, 84, 84};
    at::IntArrayRef testInputShape {4, 84, 84};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = envNum;
    option.isAtari = true;
    option.envStep = 8; //deprecated
    option.donePerEp = 1;
    option.multiLifes = false;
    //grad
    option.entropyCoef = 0.01;
    option.valueCoef = 1;
    option.maxGradNormClip = 0.5;
    option.gamma = 0.99;
    option.maxStep = 30;
    //log
    option.logInterval = 10;
    option.tensorboardLogPath = "./logs/a2crnn_testtruncpong_gae/tfevents.pb";
    //input
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    //test
    option.toTest = true;
    option.testGapEp = 100;
    option.testBatch = testClientNum;
    option.testEp = testClientNum;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./a2cngae_test0";
    //rnn
    option.hiddenNums = {3136};
    option.hidenLayerNums = {1};
    option.maxStep = 10;
    option.gruCellNum = 1;
    option.ppoLambda = 0.6;


    SoftmaxPolicy policy(outputNum);
    A2CGRUTruncGAE<AirACHOGRUNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, testEnv, policy, optimizer, option);
    a2c.train(epochNum);
}


void testPongSlimGae(const int epochNum) {
	const int batchSize = 40;
	const int envNum = 37;
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int inputNum = 4;
	const int testClientNum = 4;
//	const int maxStep = 8; //deprecated
//	const int hiddenLayerNum = 1;
//	const int hiddenNum = 256;

	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, envNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10206";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHOGRUSlimNet model(outputNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
    LOG4CXX_INFO(logger, "Model ready");


    at::IntArrayRef inputShape{4, 84, 84};
    at::IntArrayRef testInputShape {4, 84, 84};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = envNum;
    option.isAtari = true;
    option.envStep = 8; //deprecated
    option.donePerEp = 1;
    option.multiLifes = false;
    //grad
    option.entropyCoef = 0.01;
    option.valueCoef = 0.3;
    option.maxGradNormClip = 0.5;
    option.gamma = 0.99;
    option.maxStep = 30;
    //log
    option.logInterval = 10;
    option.tensorboardLogPath = "./logs/a2crnn_testpongslimgae/tfevents.pb";
    //input
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    //test
    option.toTest = true;
    option.testGapEp = 100;
    option.testBatch = testClientNum;
    option.testEp = testClientNum;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./a2cngae_test0";
    //rnn
    option.hiddenNums = {3136};
    option.hidenLayerNums = {1};
    option.maxStep = 10;
    option.gruCellNum = 1;


    SoftmaxPolicy policy(outputNum);
    A2CGRUTruncSlimGae<AirACHOGRUSlimNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, testEnv, policy, optimizer, option);
    a2c.train(epochNum);
}

void testPongSlimGae20(const int epochNum) {
	const int batchSize = 50;
	const int envNum = 47;
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int inputNum = 4;
	const int testClientNum = 4;
//	const int maxStep = 8; //deprecated
//	const int hiddenLayerNum = 1;
//	const int hiddenNum = 256;

	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, envNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHOGRUSlimNet model(outputNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
    LOG4CXX_INFO(logger, "Model ready");


    at::IntArrayRef inputShape{4, 84, 84};
    at::IntArrayRef testInputShape {4, 84, 84};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = envNum;
    option.isAtari = true;
    option.envStep = 8; //deprecated
    option.donePerEp = 1;
    option.multiLifes = false;
    //grad
    option.entropyCoef = 0.01;
    option.valueCoef = 0.3;
    option.maxGradNormClip = 0.5;
    option.gamma = 0.99;
    option.maxStep = 30;
    //log
    option.logInterval = 10;
    option.tensorboardLogPath = "./logs/a2crnn_testpongslimgae_m20/tfevents.pb";
    //input
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    //test
    option.toTest = true;
    option.testGapEp = 100;
    option.testBatch = testClientNum;
    option.testEp = testClientNum;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./a2cngae_test0";
    //rnn
    option.hiddenNums = {3136};
    option.hidenLayerNums = {1};
    option.maxStep = 20;
    option.gruCellNum = 1;


    SoftmaxPolicy policy(outputNum);
    A2CGRUTruncSlimGae<AirACHOGRUSlimNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, testEnv, policy, optimizer, option);
    a2c.train(epochNum);
}

void testPongSlimGae4(const int epochNum) {
	const int batchSize = 50;
	const int envNum = 47;
	const std::string envName = "PongNoFrameskip-v4";
	const int outputNum = 6;
	const int inputNum = 4;
	const int testClientNum = 4;
//	const int maxStep = 8; //deprecated
//	const int hiddenLayerNum = 1;
//	const int hiddenNum = 256;

	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, envNum);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10202";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHOGRUSlimNet model(outputNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
    LOG4CXX_INFO(logger, "Model ready");


    at::IntArrayRef inputShape{4, 84, 84};
    at::IntArrayRef testInputShape {4, 84, 84};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = envNum;
    option.isAtari = true;
    option.envStep = 8; //deprecated
    option.donePerEp = 1;
    option.multiLifes = false;
    //grad
    option.entropyCoef = 0.01;
    option.valueCoef = 0.3;
    option.maxGradNormClip = 0.5;
    option.gamma = 0.99;
    option.maxStep = 30;
    //log
    option.logInterval = 10;
    option.tensorboardLogPath = "./logs/a2crnn_testpongslimgae_m4/tfevents.pb";
    //input
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    //test
    option.toTest = true;
    option.testGapEp = 100;
    option.testBatch = testClientNum;
    option.testEp = testClientNum;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./a2cngae_test0";
    //rnn
    option.hiddenNums = {3136};
    option.hidenLayerNums = {1};
    option.maxStep = 4;
    option.gruCellNum = 1;


    SoftmaxPolicy policy(outputNum);
    A2CGRUTruncSlimGae<AirACHOGRUSlimNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, testEnv, policy, optimizer, option);
    a2c.train(epochNum);
}

void testBr(const int epochNum) {
	const int batchSize = 50;
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int inputNum = 4;
	const int testClientNum = 4;
//	const int maxStep = 8; //deprecated
//	const int hiddenLayerNum = 1;
//	const int hiddenNum = 256;

	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, batchSize);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10204";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHOGRUNet model(outputNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));
    LOG4CXX_INFO(logger, "Model ready");


    at::IntArrayRef inputShape{4, 84, 84};
    at::IntArrayRef testInputShape {4, 84, 84};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = batchSize;
    option.isAtari = true;
    option.envStep = 8; //deprecated
    option.donePerEp = 1;
    option.multiLifes = false;
    //grad
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.gamma = 0.99;
    option.maxStep = 30;
    //log
    option.logInterval = 10;
    option.tensorboardLogPath = "./logs/a2crnn_testtruncbr/tfevents.pb";
    //input
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    //test
    option.toTest = true;
    option.testGapEp = 100;
    option.testBatch = testClientNum;
    option.testEp = testClientNum;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./a2cngae_test0";
    //rnn
    option.hiddenNums = {3136};
    option.hidenLayerNums = {1};
    option.maxStep = 10;
    option.gruCellNum = 1;


    SoftmaxPolicy policy(outputNum);
    A2CGRUTrunc<AirACHOGRUNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, testEnv, policy, optimizer, option);
    a2c.train(epochNum);
}

void testBrGae(const int epochNum) {
	const int batchSize = 50;
	const std::string envName = "BreakoutNoFrameskip-v4";
	const int outputNum = 4;
	const int inputNum = 4;
	const int testClientNum = 4;
//	const int maxStep = 8; //deprecated
//	const int hiddenLayerNum = 1;
//	const int hiddenNum = 256;

	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, batchSize);
	env.init();
	std::string testServerAddr = "tcp://127.0.0.1:10206";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	AirACHOGRUNet model(outputNum);
	model.to(deviceType);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(5e-5));
    LOG4CXX_INFO(logger, "Model ready");


    at::IntArrayRef inputShape{4, 84, 84};
    at::IntArrayRef testInputShape {4, 84, 84};
    DqnOption option(inputShape, testInputShape, deviceType);
    //env
    option.envNum = batchSize;
    option.isAtari = true;
    option.envStep = 8; //deprecated
    option.donePerEp = 5;
    option.multiLifes = true;
    //grad
    option.entropyCoef = 0.002;
    option.valueCoef = 0.3;
    option.maxGradNormClip = 0.5;
    option.gamma = 0.99;
    option.maxStep = 30;
    //log
    option.logInterval = 10;
    option.tensorboardLogPath = "./logs/a2crnn_testtruncbrgae/tfevents.pb";
    //input
    option.inputScale = 255;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    //test
    option.toTest = true;
    option.testGapEp = 100;
    option.testBatch = testClientNum;
    option.testEp = testClientNum;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./a2cngae_test0";
    //rnn
    option.hiddenNums = {3136};
    option.hidenLayerNums = {1};
    option.maxStep = 10;
    option.gruCellNum = 1;


    SoftmaxPolicy policy(outputNum);
    A2CGRUTruncGAE<AirACHOGRUNet, AirEnv, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, testEnv, policy, optimizer, option);
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

//	testCart(atoi(argv[1]));
//	testPong(atoi(argv[1]));
//	testCartGae(atoi(argv[1]));
//	testPongGae(atoi(argv[1]));
	testPongSlimGae20(atoi(argv[1]));
//	testBr(atoi(argv[1]));
//	testBrGae(atoi(argv[1]));
//	testCartSlim(atoi(argv[1]));
//	testPongSlim(atoi(argv[1]));


	return 0;
}


