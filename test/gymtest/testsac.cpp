/*
 * testsac.cpp
 *
 *  Created on: Sep 17, 2021
 *      Author: zf
 */





#include "alg/sac.hpp"
#include "alg/saczip.hpp"
#include "alg/sacnopolicy.hpp"

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
#include "gymtest/lunarnets/cartsacqnet.h"
#include "gymtest/lunarnets/cartsacpolicynet.h"
#include "gymtest/airnets/airachonet.h"
#include "gymtest/airnets/airsacqnet.h"
#include "gymtest/airnets/airsacpnet.h"
#include "gymtest/noisynets/noisycartfcnet.h"
#include "gymtest/noisynets/noisyaircnnnet.h"
#include "gymtest/train/rawpolicy.h"
#include "gymtest/train/softmaxpolicy.h"
#include "probeenvs/ProbeEnvWrapper.h"

#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/simplelayout.h>
#include <log4cxx/logmanager.h>

#include <vector>
#include <cmath>
#include "alg/utils/dqnoption.h"

namespace {
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("sactest"));
const torch::Device deviceType = torch::kCUDA;

void testProbe(const int epochNum) {
	const int batchSize = 1;
	const int inputNum = 4;
	const int envId = 5;
	const int outputNum = 2;

	const int envNum = batchSize;
	ProbeEnvWrapper env(inputNum, envId, envNum);

	CartSacQNet model1(inputNum, outputNum);
	model1.to(deviceType);
	CartSacQNet model2(inputNum, outputNum);
	model2.to(deviceType);
	CartSacQNet targetModel1(inputNum, outputNum);
	targetModel1.to(deviceType);
	CartSacQNet targetModel2(inputNum, outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	CartSacPolicyNet pModel(inputNum, outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(0.005));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(0.005));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(0.005));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(1e-4));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{envNum, 4};
    DqnOption option(inputShape, deviceType);
    option.envNum = envNum;
    //target model
    option.targetUpdateStep = 100;
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
    option.gamma = 0.95;
    //grad
    option.batchSize = 4;
    option.startStep = 50;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testprobe";
    //model
    option.saveModel = false;
    option.savePathPrefix = "./sac_testprobe";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = false;
    //sac
    option.targetEntropy = -0.98 * std::log(1.0f / (float)outputNum);

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    Sac<CartSacQNet, CartSacPolicyNet, ProbeEnvWrapper, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				env,
				deviceType, option);

    sac.train(epochNum);
//    dqn.train(epochNum);
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

	std::string testServerAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	LunarEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	CartSacQNet model1(inputNum, outputNum);
	model1.to(deviceType);
	CartSacQNet model2(inputNum, outputNum);
	model2.to(deviceType);
	CartSacQNet targetModel1(inputNum, outputNum);
	targetModel1.to(deviceType);
	CartSacQNet targetModel2(inputNum, outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	CartSacPolicyNet pModel(inputNum, outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	float rl = 0.003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(rl));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(rl));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(rl));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(rl)); //rl: 46, rl / 2: 349, rl * 2: > 350
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 10000;
    //explore
//    option.exploreBegin = 0;
//    option.exploreEnd = 0;
//    option.explorePart = 1;
    //input
    option.inputScale = 1;
    option.rewardScale = 1;
    option.rewardMin = -1000; //TODO: reward may not require clip
    option.rewardMax = 1000;
    option.gamma = 0.98;
    //grad
    option.batchSize = 4;
    option.startStep = 50;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testcart";
    option.logInterval = 100;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./sac_testcart";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = 1000;
	option.testEp = 8;
    //sac
    option.targetEntropy = -0.98 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    Sac<CartSacQNet, CartSacPolicyNet, LunarEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}

void testtestCart(const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 1; //8
	const int outputNum = 2;
	const int inputNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	CartSacQNet model1(inputNum, outputNum);
	model1.to(deviceType);
	CartSacQNet model2(inputNum, outputNum);
	model2.to(deviceType);
	CartSacQNet targetModel1(inputNum, outputNum);
	targetModel1.to(deviceType);
	CartSacQNet targetModel2(inputNum, outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	CartSacPolicyNet pModel(inputNum, outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(0.005));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(0.005));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(0.005));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(1e-4));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 1000;
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
    option.gamma = 0.95;
    //grad
    option.batchSize = 4;
    option.startStep = 50;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testcart";
    //model
    option.saveModel = false;
    option.savePathPrefix = "./sac_testcart";
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/sac_testcart";
    //sac
    option.targetEntropy = -0.98 * std::log(1.0f / (float)outputNum);

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    Sac<CartSacQNet, CartSacPolicyNet, LunarEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				env,
				deviceType, option);
    sac.test(epochNum);
}

void testLog(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float rl = 3e-4;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(rl));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(rl));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(rl));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(rl));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 80000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    //TODO: inputScale = 255;
    option.inputScale = 255;
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
    option.statPathPrefix = "./sac_testlog";
    option.logInterval = 100;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./sac_testlog";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 8;
    //sac
    option.targetEntropy = -0.98 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    Sac<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}


void testPacman(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 80000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -100; //TODO: reward may not require clip
    option.rewardMax = 100;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 8;
    //sac
    option.targetEntropy = -0.98 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    Sac<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}


void testPacman1(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 80000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 2;
    option.rewardMin = -100; //TODO: reward may not require clip
    option.rewardMax = 100;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman1";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman1";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 8;
    //sac
    option.targetEntropy = -0.98 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    Sac<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}


void testPacman2(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 80000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 1.5;
    option.rewardMin = -100; //TODO: reward may not require clip
    option.rewardMax = 100;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman2";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman2";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 8;
    //sac
    option.targetEntropy = -0.98 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    Sac<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}


void testPacman3(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 80000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -100; //TODO: reward may not require clip
    option.rewardMax = 100;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman3";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman3";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/sac_testpacman";
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 8;
    //sac
    option.targetEntropy = -0.98 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    Sac<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}

void testPacman4(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 80000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -100; //TODO: reward may not require clip
    option.rewardMax = 100;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman4";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman4";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 8;
    //sac
    option.targetEntropy = -0.98 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    Sac<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}


void testPacman5(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 100000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 0.5;
    option.rewardMin = -100; //TODO: reward may not require clip
    option.rewardMax = 100;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman5";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman5";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 8;
    //sac
    option.targetEntropy = -0.98 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    Sac<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}


void testPacman6(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 100000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman6";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman6";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 8;
    //sac
    option.targetEntropy = -0.98 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    Sac<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}


void testPacman7(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 100000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman7";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman7";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/sac_testpacman6";
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 8;
    //sac
    option.targetEntropy = -0.98 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    Sac<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}


void testPacman8(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 100000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman8";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman8";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/sac_testpacman7";
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 8;
    //sac
    option.targetEntropy = -0.98 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    Sac<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}

void testPacman9(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 100000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -0.5; //TODO: reward may not require clip
    option.rewardMax = 0.5;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman9";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman9";
    option.loadModel = false;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/sac_testpacman7";
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 8;
    //sac
    option.targetEntropy = -0.98 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    Sac<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}


void testPacman10(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 100000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 0.5;
    option.rewardMin = -2; //TODO: reward may not require clip
    option.rewardMax = 2;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman10";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman10";
    option.loadModel = false;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/sac_testpacman7";
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 8;
    //sac
    option.targetEntropy = -0.98 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    Sac<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}


void testPacman11(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 10000;
    option.tau = 1;
    //buffer
    option.rbCap = 200000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -100; //TODO: reward may not require clip
    option.rewardMax = 100;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman11";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman11";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = option.targetUpdateStep * 1;
	option.testEp = 4;
	option.testBatch = 1;
    //sac
    option.targetEntropy = -0.85 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    SacZip<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}

void testPacman12(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 10000;
    option.tau = 1;
    //buffer
    option.rbCap = 200000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -100; //TODO: reward may not require clip
    option.rewardMax = 100;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman12";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman12";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = option.targetUpdateStep * 1;
	option.testEp = 4;
	option.testBatch = 1;
    //sac
    option.targetEntropy = -0.85 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    SacNoPolicy<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}


void testPacman13(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 300000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman13";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman13";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = option.targetUpdateStep * 1;
	option.testEp = 4;
	option.testBatch = 1;
    //sac
    option.targetEntropy = -0.98 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    SacZip<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}


void testPacman14(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 300000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 0.5;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman14";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman14";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = option.targetUpdateStep * 1;
	option.testEp = 4;
	option.testBatch = 1;
    //sac
    option.targetEntropy = -0.98 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    SacZip<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}


void testPacman15(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 300000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman15";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman15";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = option.targetUpdateStep * 1;
	option.testEp = 4;
	option.testBatch = 1;
    //sac
    option.targetEntropy = -0.75 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    SacZip<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}

void testPacman16(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 300000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman16";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman16";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = option.targetUpdateStep * 1;
	option.testEp = 4;
	option.testBatch = 1;
    //sac
    option.targetEntropy = -0.9 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    SacZip<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}


void testPacman17(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 300000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 10;
    option.rewardMin = -10; //TODO: reward may not require clip
    option.rewardMax = 10;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman17";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman17";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = option.targetUpdateStep * 1;
	option.testEp = 4;
	option.testBatch = 1;
    //sac
    option.targetEntropy = -0.95 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    SacZip<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}



void testPacman18(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 300000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 5;
    option.rewardMin = -5; //TODO: reward may not require clip
    option.rewardMax = 5;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman18";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman18";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = option.targetUpdateStep * 1;
	option.testEp = 4;
	option.testBatch = 1;
    //sac
    option.targetEntropy = -0.95 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    SacZip<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}


void testPacman19(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 300000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 10;
    option.rewardMin = -10; //TODO: reward may not require clip
    option.rewardMax = 10;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman19";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman19";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/sac_testpacman17";

    //test
    option.toTest = true;
	option.testGapEp = option.targetUpdateStep * 1;
	option.testEp = 4;
	option.testBatch = 1;
    //sac
    option.targetEntropy = -0.95 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    SacZip<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}



void testPacman20(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 300000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 8;
    option.rewardMin = -8; //TODO: reward may not require clip
    option.rewardMax = 8;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman20";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman20";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = option.targetUpdateStep * 1;
	option.testEp = 4;
	option.testBatch = 1;
    //sac
    option.targetEntropy = -0.95 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    SacZip<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}

//TODO: Try less entropy coef for reward * 10 case


void testPacman21(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 300000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 10;
    option.rewardMin = -10; //TODO: reward may not require clip
    option.rewardMax = 10;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman21";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman21";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/sac_testpacman17";

    //test
    option.toTest = true;
	option.testGapEp = option.targetUpdateStep * 1;
	option.testEp = 4;
	option.testBatch = 1;
    //sac
    option.targetEntropy = -0.90 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    SacZip<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}



void testPacman22(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 300000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 10;
    option.rewardMin = -10; //TODO: reward may not require clip
    option.rewardMax = 10;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman22";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman22";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/sac_testpacman17";

    //test
    option.toTest = true;
	option.testGapEp = option.targetUpdateStep * 1;
	option.testEp = 4;
	option.testBatch = 1;
    //sac
    option.targetEntropy = -0.98 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    SacZip<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}

//clip alpha
void testPacman23(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 300000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 10;
    option.rewardMin = -10; //TODO: reward may not require clip
    option.rewardMax = 10;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman23";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman23";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/sac_testpacman17";

    //test
    option.toTest = true;
	option.testGapEp = option.targetUpdateStep * 1;
	option.testEp = 4;
	option.testBatch = 1;
    //sac
    option.targetEntropy = -0.95 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    SacZip<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}

//clip alpha
void testPacman24(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 300000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 10;
    option.rewardMin = -10; //TODO: reward may not require clip
    option.rewardMax = 10;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman24";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman24";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/sac_testpacman23";

    //test
    option.toTest = true;
	option.testGapEp = option.targetUpdateStep * 1;
	option.testEp = 4;
	option.testBatch = 1;
    //sac
    option.targetEntropy = -0.95 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    SacZip<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}

//clip alpha
void testPacman25(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float lr = 0.0003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(lr));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(lr));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 300000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 10;
    option.rewardMin = -10; //TODO: reward may not require clip
    option.rewardMax = 10;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpacman25";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testpacman25";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/sac_testpacman24";

    //test
    option.toTest = true;
	option.testGapEp = option.targetUpdateStep * 1;
	option.testEp = 4;
	option.testBatch = 1;
    //sac
    option.targetEntropy = -0.95 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    SacZip<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}




void testtestPacman(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10210";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

//	std::string testServerAddr = "tcp://127.0.0.1:10201";
//	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
//	AirEnv testEnv(testServerAddr, envName, clientNum);
//	testEnv.init();
//	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(0.0005));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(0.0005));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(0.0005));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(0.0005));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    //TODO: increase update gap
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 128;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 32;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testtestpacman2";
    //model
    option.saveModel = false;
    option.savePathPrefix = "./sac_testtestpacman2";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/sac_testpacman2";
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 8;
    //sac
    option.targetEntropy = -0.98 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    Sac<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				env,
				deviceType, option);

    sac.test(epochNum, true);
}


void testAssault(const int epochNum) {
	const std::string envName = "AssaultNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 7;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
	float rl = 3e-4;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(rl));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(rl));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(rl));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(rl));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 80000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testassault";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testassault";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 8;
    //sac
    option.targetEntropy = -0.98 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    Sac<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}


void testAssault2(const int epochNum) {
	const std::string envName = "AssaultNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 7;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(0.0005));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(0.0005));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(0.0005));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(0.0005));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 100000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testassault2";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./sac_testassault2";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/sac_testassault";
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 8;
    //sac
    option.targetEntropy = -0.98 * std::log(1.0f / (float)outputNum);
    //TODO: decrease envStep into 1
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    Sac<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}


void testPong(const int epochNum) {
	const std::string envName = "PongNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 6;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, clientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(0.0005));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(0.0005));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(0.0005));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(0.0005));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 100000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testpong";
    option.logInterval = 100;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./sac_testpong";
    option.loadModel = false;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/sac_testassault";
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 8;
	option.testBatch = 1;
    //sac
    option.targetEntropy = -0.98 * std::log(1.0f / (float)outputNum);
    //TODO: decrease envStep into 1
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    SacZip<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}



void testtestAssault(const int epochNum) {
	const std::string envName = "AssaultNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 7;
	std::string serverAddr = "tcp://127.0.0.1:10210";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

//	std::string testServerAddr = "tcp://127.0.0.1:10201";
//	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
//	AirEnv testEnv(testServerAddr, envName, clientNum);
//	testEnv.init();
//	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	AirSacQNet model1(outputNum);
	model1.to(deviceType);
	AirSacQNet model2(outputNum);
	model2.to(deviceType);
	AirSacQNet targetModel1(outputNum);
	targetModel1.to(deviceType);
	AirSacQNet targetModel2(outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	AirSacPNet pModel(outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	//TODO: decrease lr
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(0.0005));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(0.0005));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(0.0005));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(0.0005));
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4, 84, 84};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 8000;
    option.tau = 1;
    //buffer
    option.rbCap = 128;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 1;
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.99;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./sac_testtestassault";
    //model
    option.saveModel = false;
    option.savePathPrefix = "./sac_testtestassault";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/sac_testassault";
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 8;
    //sac
    option.targetEntropy = -0.98 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    Sac<AirSacQNet, AirSacPNet, AirEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				env,
				deviceType, option);

    sac.test(epochNum, true);
}

void testNoPolicyCart(const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 1; //8
	const int testClientNum = 1;
	const int outputNum = 2;
	const int inputNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	LunarEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	CartSacQNet model1(inputNum, outputNum);
	model1.to(deviceType);
	CartSacQNet model2(inputNum, outputNum);
	model2.to(deviceType);
	CartSacQNet targetModel1(inputNum, outputNum);
	targetModel1.to(deviceType);
	CartSacQNet targetModel2(inputNum, outputNum);
	targetModel2.to(deviceType);
	LOG4CXX_INFO(logger, "Q models ready");

	CartSacPolicyNet pModel(inputNum, outputNum);
	pModel.to(deviceType);
	LOG4CXX_INFO(logger, "Policy model ready");

	torch::Tensor logAlpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(deviceType));
	LOG4CXX_INFO(logger, "Alpha ready");

	float rl = 0.003;
    torch::optim::Adam optimizerQ1(model1.parameters(), torch::optim::AdamOptions(rl));
    torch::optim::Adam optimizerQ2(model2.parameters(), torch::optim::AdamOptions(rl));
    LOG4CXX_INFO(logger, "Q optimizer ready");
    torch::optim::Adam optimizerP(pModel.parameters(), torch::optim::AdamOptions(rl));
    torch::optim::Adam optimizerA({logAlpha}, torch::optim::AdamOptions(rl)); //rl: 46, rl / 2: 349, rl * 2: > 350
    LOG4CXX_INFO(logger, "Model ready");



    at::IntArrayRef inputShape{clientNum, 4};
    DqnOption option(inputShape, deviceType);
    option.envNum = clientNum;
    //target model
    option.targetUpdateStep = 1000;
    option.tau = 1;
    //buffer
    option.rbCap = 10000;
    //explore
//    option.exploreBegin = 0;
//    option.exploreEnd = 0;
//    option.explorePart = 1;
    //input
    option.inputScale = 1;
    option.rewardScale = 1;
    option.rewardMin = -255; //TODO: reward may not require clip
    option.rewardMax = 255;
    option.gamma = 0.98;
    //grad
    option.batchSize = 4;
    option.startStep = 50;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.statPathPrefix = "./saczip_testcart";
    option.logInterval = 100;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./saczip_testcart";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
    option.testGapEp = 1000;
    option.testEp = 4;
    option.testBatch = testClientNum;
    //sac
    option.targetEntropy = -0.75 * std::log(1.0f / (float)outputNum);
    option.envStep = 4;

    SoftmaxPolicy policy(outputNum);

    /*
     * Sac<QNetType, PNetType, EnvType, PolicyType, QOptimizerType, POptimizerType, AlphaOptimizerType>::Sac(
		QNetType& qModel1, QNetType& qModel2, QNetType& qTargetModel1, QNetType& qTargetModel2, QOptimizerType& qOpt1, QOptimizerType& qOpt2,
		PNetType& pModel, POptimizerType& pOpt,
		torch::Tensor& iLogAlpha, AlphaOptimizerType& aOpt,
		EnvType& iEnv, PolicyType iPolicy,
		const torch::Device dType, DqnOption iOption)
     */
    SacNoPolicy<CartSacQNet, CartSacPolicyNet, LunarEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
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
//	testPong1(atoi(argv[1]));
//	testBreakout(atoi(argv[1]));

	testCart(atoi(argv[1]));
//	testProbe(atoi(argv[1]));
//	testLog(atoi(argv[1]));
//	testPacman25(atoi(argv[1]));
//	testPong(atoi(argv[1]));
//	testtestCart(atoi(argv[1]));
//	testAssault2(atoi(argv[1]));

//	testZipCart(atoi(argv[1]));
//	testNoPolicyCart(atoi(argv[1]));

//	testtestPacman(atoi(argv[1]));
//	testtestAssault(atoi(argv[1]));



	LOG4CXX_INFO(logger, "End of test");
}

