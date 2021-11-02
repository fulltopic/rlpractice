/*
 * testsaczip.cpp
 *
 *  Created on: Oct 12, 2021
 *      Author: zf
 */




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
#include "alg/dqnoption.h"

#include "probeenvs/ProbeEnvWrapper.h"

#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/simplelayout.h>
#include <log4cxx/logmanager.h>

#include <vector>
#include <cmath>

namespace {
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("sacziptest"));
const torch::Device deviceType = torch::kCUDA;

//Zip is not good for Cart as input is of float instead of int
void testZipCart(const int epochNum) {
	const std::string envName = "CartPole-v0";
	const int clientNum = 1; //8
	const int testClientNum = 1;
	const int outputNum = 2;
	const int inputNum = 4;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	LunarEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10203";
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
    option.rewardMin = -1; //TODO: reward may not require clip
    option.rewardMax = 1;
    option.gamma = 0.98;
    //grad
    option.batchSize = 4;
    option.startStep = 50;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
//    option.statPathPrefix = "./saczip_testcart";
    option.tensorboardLogPath = "./logs/saczip_testcart/tfevents.pb";
    option.logInterval = 100;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./saczip_testcart";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = false;
    option.testGapEp = 1000;
    option.testEp = 4;
    option.testBatch = testClientNum;
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
    SacZip<CartSacQNet, CartSacPolicyNet, LunarEnv, SoftmaxPolicy, torch::optim::Adam, torch::optim::Adam, torch::optim::Adam>
    	sac(model1, model2, targetModel1, targetModel2, optimizerQ1, optimizerQ2,
    			pModel, optimizerP,
				logAlpha, optimizerA,
				env, policy,
				testEnv,
				deviceType, option);

    sac.train(epochNum);
}

void testPacman0(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int testClientNum = 1;
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	torch::manual_seed(0);

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
    //output
    option.outputNum = outputNum;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1000;
    //log
    option.statCap = 128;
    option.tensorboardLogPath = "./logs/saczip_testpac0/tfevents.pb";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./saczip_testpacman0";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 3;
	option.testBatch = testClientNum;
	option.livePerEpisode = 3;
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

void testPacman01(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 4; //8
	const int testClientNum = 9;
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	torch::manual_seed(0);

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
    option.targetUpdateStep = 6000;
    option.tau = 1;
    //buffer
    option.rbCap = 300000;
    //explore
    option.exploreBegin = 0;
    option.exploreEnd = 0;
    option.explorePart = 1;
    //input
    option.inputScale = 255;
    option.rewardScale = 2;
    option.rewardMin = -2; //TODO: reward may not require clip
    option.rewardMax = 2;
    option.gamma = 0.99;
    //output
    option.outputNum = outputNum;
    //grad
    option.batchSize = 64;
    option.startStep = 5000;
    option.maxGradNormClip = 1000;
    //log
    option.statCap = 128;
    option.tensorboardLogPath = "./logs/saczip_testpac01/tfevents.pb";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./saczip_testpacman01";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 9;
	option.testBatch = testClientNum;
	option.livePerEpisode = 3;
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


void testPacman1(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int testClientNum = 9;
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	torch::manual_seed(0);

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
    //output
    option.outputNum = outputNum;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1000;
    //log
    option.statCap = 128;
    option.tensorboardLogPath = "./logs/saczip_testpac1/tfevents.pb";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./saczip_testpacman1";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 18;
	option.testBatch = testClientNum;
	option.livePerEpisode = 3;
    //sac
	option.fixedEntropy = true;
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


void testPacman11(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int testClientNum = 9;
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	torch::manual_seed(0);

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
    option.rewardScale = 2;
    option.rewardMin = -2; //TODO: reward may not require clip
    option.rewardMax = 2;
    option.gamma = 0.99;
    //output
    option.outputNum = outputNum;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1000;
    //log
    option.statCap = 128;
    option.tensorboardLogPath = "./logs/saczip_testpac11/tfevents.pb";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./saczip_testpacman11";
    option.saveThreshold = 1000;
    option.saveStep = 100;
    option.loadModel = false;
    option.loadOptimizer = false;

    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 18;
	option.testBatch = testClientNum;
	option.livePerEpisode = 3;
    //sac
	option.fixedEntropy = true;
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

void testPacman2(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int testClientNum = 9;
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	torch::manual_seed(0);

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
    //output
    option.outputNum = outputNum;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1000;
    //log
    option.statCap = 128;
    option.tensorboardLogPath = "./logs/saczip_testpac2/tfevents.pb";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./saczip_testpacman2";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 18;
	option.testBatch = testClientNum;
	option.livePerEpisode = 3;
    //sac
	option.fixedEntropy = true;
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



void testPacman3(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int testClientNum = 9;
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	torch::manual_seed(0);

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
    //output
    option.outputNum = outputNum;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1000;
    //log
    option.statCap = 128;
    option.tensorboardLogPath = "./logs/saczip_testpac3/tfevents.pb";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./saczip_testpacman3";
    option.saveThreshold = 1000;
    option.saveStep = 100;
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 18;
	option.testBatch = testClientNum;
	option.livePerEpisode = 3;
    //sac
	option.fixedEntropy = true;
    option.targetEntropy = -0.80 * std::log(1.0f / (float)outputNum);
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


void testPacman4(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int testClientNum = 8;
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
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
    //output
    option.outputNum = outputNum;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1000;
    //log
    option.statCap = 128;
    option.tensorboardLogPath = "./logs/saczip_testpac4/tfevents.pb";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./saczip_testpacman4";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/saczip_testpacman3";
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 40;
	option.testBatch = testClientNum;
	option.testRender = false;
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


void testPacman5(const int epochNum) {
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
    //output
    option.outputNum = outputNum;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1000;
    //log
    option.statCap = 128;
    option.tensorboardLogPath = "./logs/saczip_testpac5/tfevents.pb";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./saczip_testpacman5";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/saczip_testpacman0";
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 10000;
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

void testPacman6(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int testClientNum = 8;
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	torch::manual_seed(0);

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
    option.rewardMin = -100; //TODO: reward may not require clip
    option.rewardMax = 100;
    option.gamma = 0.99;
    //output
    option.outputNum = outputNum;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1000;
    //log
    option.statCap = 128;
    option.tensorboardLogPath = "./logs/saczip_testpac6/tfevents.pb";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./saczip_testpacman6";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 40;
	option.testBatch = testClientNum;
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


void testPacman7(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int testClientNum = 1;
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	torch::manual_seed(0);

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
    //output
    option.outputNum = outputNum;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1000;
    //log
    option.statCap = 128;
    option.tensorboardLogPath = "./logs/saczip_testpac7/tfevents.pb";
    option.logInterval = 100;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./saczip_testpacman7";
    option.saveThreshold = 1000;
    option.saveStep = 50;
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 3;
	option.testBatch = testClientNum;
	option.livePerEpisode = 3;
    //sac
	option.fixedEntropy = false;
	option.targetSteps = {0, 600000, 1000000};
	option.targetEntropies = { -0.98 * std::log(1.0f / (float)outputNum),  -0.95 * std::log(1.0f / (float)outputNum),  -0.90 * std::log(1.0f / (float)outputNum)};
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

void testQbert0(const int epochNum) {
	const std::string envName = "QbertNoFrameskip-v4";
	const int clientNum = 8; //8
	const int testClientNum = 4;
	const int outputNum = 6;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
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
    //output
    option.outputNum = outputNum;
    //grad
    option.batchSize = 64;
    option.startStep = 5000;
    option.maxGradNormClip = 1000;
    //log
    option.statCap = 128;
    option.tensorboardLogPath = "./logs/saczip_testqbert0/tfevents.pb";
    option.logInterval = 1000;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./saczip_testqbert0";
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 8;
	option.testBatch = testClientNum;
	option.livePerEpisode = 4;
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

void testQbert1(const int epochNum) {
	const std::string envName = "QbertNoFrameskip-v4";
	const int clientNum = 1; //8
	const int testClientNum = 4;
	const int outputNum = 6;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
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
    //output
    option.outputNum = outputNum;
    //grad
    option.batchSize = 64;
    option.startStep = 5000;
    option.maxGradNormClip = 1000;
    //log
    option.statCap = 128;
    option.tensorboardLogPath = "./logs/saczip_testqbert1/tfevents.pb";
    option.logInterval = 1000;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./saczip_testqbert1";
    option.saveThreshold = 500;
    option.saveStep = 100;
    option.loadModel = false;
    option.loadOptimizer = false;
    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 8;
	option.testBatch = testClientNum;
	option.livePerEpisode = 4;
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

void testQbert11(const int epochNum) {
	const std::string envName = "QbertNoFrameskip-v4";
	const int clientNum = 1; //8
	const int testClientNum = 4;
	const int outputNum = 6;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
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
    //output
    option.outputNum = outputNum;
    //grad
    option.batchSize = 64;
    option.startStep = 5000;
    option.maxGradNormClip = 1000;
    //log
    option.statCap = 128;
    option.tensorboardLogPath = "./logs/saczip_testqbert11/tfevents.pb";
    option.logInterval = 1000;
    //model
    option.saveModel = true;
    option.savePathPrefix = "./saczip_testqbert11";
    option.saveThreshold = 500;
    option.saveStep = 100;
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/saczip_testqbert1";

    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 8;
	option.testBatch = testClientNum;
	option.livePerEpisode = 4;
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

void testtestPacman(const int epochNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 9;
	const int testClientNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
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
    option.rbCap = 3000;
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
    //output
    option.outputNum = outputNum;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.tensorboardLogPath = "./logs/saczip_testtestpac/tfevents.pb";
    option.logInterval = 100;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./saczip_testtestpac";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/saczip_testpacman7";

    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 100;
	option.testBatch = testClientNum;
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

    sac.test(epochNum, true);
}


void testtestQbert(const int epochNum) {
	const std::string envName = "QbertNoFrameskip-v4";
	const int clientNum = 1; //8
	const int outputNum = 6;
	const int testClientNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10205";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10207";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
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
    option.rbCap = 3000;
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
    //output
    option.outputNum = outputNum;
    //grad
    option.batchSize = 64;
    option.startStep = 10000;
    option.maxGradNormClip = 1;
    //log
    option.statCap = 128;
    option.tensorboardLogPath = "./logs/saczip_testtestqbert/tfevents.pb";
    option.logInterval = 100;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./saczip_testtestpac";
    option.loadModel = true;
    option.loadOptimizer = true;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/saczip_testqbert0";

    //test
    option.toTest = true;
	option.testGapEp = 10000;
	option.testEp = 100;
	option.testBatch = testClientNum;
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

    sac.test(epochNum, true);
}


void testLoad(int stepNum) {
	const std::string envName = "MsPacmanNoFrameskip-v4";
	const int clientNum = 1; //8
	const int testClientNum = 1;
	const int outputNum = 9;
	std::string serverAddr = "tcp://127.0.0.1:10203";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, envName, clientNum);
	env.init();
	LOG4CXX_INFO(logger, "Env " << envName << " ready");

	std::string testServerAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << testServerAddr);
	AirEnv testEnv(testServerAddr, envName, testClientNum);
	testEnv.init();
	LOG4CXX_INFO(logger, "Test env " << envName << " ready");

	torch::manual_seed(0);

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
    option.rbCap = 100;
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
    //output
    option.outputNum = outputNum;
    //grad
    option.batchSize = 64;
    option.startStep = 20000;
    option.maxGradNormClip = 1000;
    //log
    option.statCap = 128;
    option.tensorboardLogPath = "./logs/saczip_testlog/tfevents.pb";
    option.logInterval = 100;
    //model
    option.saveModel = false;
    option.savePathPrefix = "./saczip_testpacmanload";
    option.saveThreshold = 1000;
    option.saveStep = 50;
    option.loadModel = true;
    option.loadOptimizer = false;
    option.loadPathPrefix = "/home/zf/workspaces/workspace_cpp/rlpractice/build/test/gymtest/saczip_testpacman0";
    //test
    option.toTest = false;
	option.testGapEp = 10000;
	option.testEp = 3;
	option.testBatch = testClientNum;
	option.livePerEpisode = 3;
    //sac
	option.fixedEntropy = false;
	option.targetSteps = {0, 600000, 1000000};
	option.targetEntropies = { -0.98 * std::log(1.0f / (float)outputNum),  -0.95 * std::log(1.0f / (float)outputNum),  -0.90 * std::log(1.0f / (float)outputNum)};
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

//    sac.testLoad(stepNum);
    sac.test(1, false);
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

//	testProbe(atoi(argv[1]));
//	testLog(atoi(argv[1]));
//	testQbert11(atoi(argv[1]));
	testPacman01(atoi(argv[1]));
//	testPong(atoi(argv[1]));
//	testtestCart(atoi(argv[1]));
//	testAssault2(atoi(argv[1]));

//	testZipCart(atoi(argv[1]));
//	testNoPolicyCart(atoi(argv[1]));

//	testtestPacman(atoi(argv[1]));
//	testtestQbert(atoi(argv[1]));
//	testtestAssault(atoi(argv[1]));
//	testLoad(atoi(argv[1]));


	LOG4CXX_INFO(logger, "End of test");
}
