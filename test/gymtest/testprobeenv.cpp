/*
 * testprobeenv.cpp
 *
 *  Created on: Aug 8, 2021
 *      Author: zf
 */


#include "probeenvs/ProbeEnvWrapper.h"

#include "alg/cnn/a2cnstep.hpp"

#include "alg/cnn/pporandom.hpp"

#include "gymtest/env/airenv.h"
#include "gymtest/env/lunarenv.h"
#include "gymtest/cnnnets/airnets/aircnnnet.h"
#include "gymtest/cnnnets/airnets/airacbmnet.h"
#include "gymtest/cnnnets/airnets/airacnet.h"
#include "gymtest/cnnnets/airnets/airacbmsmallkernelnet.h"
#include "gymtest/cnnnets/lunarnets/cartacnet.h"
#include "gymtest/cnnnets/airnets/airachonet.h"
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
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("probenv"));
const torch::Device deviceType = torch::kCUDA;

void test0(const int envId, const int outputNum, const int epochNum) {
	const int batchSize = 6;
	const int inputNum = 4;

	const int num = batchSize;
	ProbeEnvWrapper env(inputNum, envId, num);

	CartACFcNet model(inputNum, outputNum);
	model.to(deviceType);
//	targetModel.to(deviceType);
//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.00025).eps(0.01).alpha(0.95));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));
//	torch::optim::RMSprop optimizer(model.parameters());
    LOG4CXX_INFO(logger, "Model ready");

    at::IntArrayRef inputShape{num, 4};
    DqnOption option(inputShape, deviceType);
    option.isAtari = false;
    option.donePerEp = 1;
    option.multiLifes = false;
    option.statCap = batchSize * 2;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.ppoLambda = 0.95;
    option.gamma = 0.99;
    option.statPathPrefix = "./probe_test0";
    option.saveModel = false;
    option.savePathPrefix = "./probe_test0";
    option.toTest = false;
    option.inputScale = 1;
    option.batchSize = batchSize;
    option.rewardScale = 1;
    option.rewardMin = -1;
    option.rewardMax = 1;
    option.valueClip = false;
    option.normReward = false;
    option.loadModel = false;
    option.loadOptimizer = false;

    SoftmaxPolicy policy(outputNum);
    const int maxStep = 4;
    A2CNStep<CartACFcNet, ProbeEnvWrapper, SoftmaxPolicy, torch::optim::Adam> a2c(model, env, env, policy, optimizer, maxStep, option);
    a2c.train(epochNum);
}


void test1(const int envId, const int outputNum, const int epochNum) {
	const int clientNum = 3; //8
	const int inputNum = 4;

	ProbeEnvWrapper env(inputNum, envId, clientNum);

	CartACFcNet model(inputNum, outputNum);
	model.to(deviceType);

//    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-2)); //rmsprop: 0.00025
//    torch::optim::RMSprop optimizer(model.parameters(),
//    		torch::optim::RMSpropOptions(7e-4).eps(1e-5).alpha(0.99));
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//	torch::optim::RMSprop optimizer(model.parameters());
//    RawPolicy policy(1, outputNum);
    LOG4CXX_INFO(logger, "Model ready");

    const int maxStep = 3; //from stat, seemed a reward feedback at first 25 steps
    const int roundNum = 2;

    at::IntArrayRef inputShape{clientNum, 4};
    DqnOption option(inputShape, deviceType, 4096, 0.99);
    option.isAtari = false;
    option.entropyCoef = 0.01;
    option.valueCoef = 0.5;
    option.maxGradNormClip = 0.5;
    option.statPathPrefix = "./probe_test1";
    option.saveModel = false;
    option.savePathPrefix = "./probe_test1";
    option.toTest = false;
    option.inputScale = 1;
    option.batchSize = maxStep;
    option.envNum = clientNum;
    option.epochNum = 8; //4
    option.trajStepNum = maxStep * roundNum; //200 //TODO:
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
    PPORandom<CartACFcNet, ProbeEnvWrapper, SoftmaxPolicy, torch::optim::Adam> ppo(model, env, env, policy, optimizer, option, outputNum);
    ppo.train(epochNum);
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

	test1(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
//	test18(atoi(argv[1]));

//Test
//	testtest204(atoi(argv[1]));

//GAE
//	test102(atoi(argv[1]));

//Clipped
//	test205(atoi(argv[1]));

//	testtest0(atoi(argv[1]), atoi(argv[2]));

	LOG4CXX_INFO(logger, "End of test");
}
