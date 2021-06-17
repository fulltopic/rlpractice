/*
 * testrawdqn.cpp
 *
 *  Created on: Apr 5, 2021
 *      Author: zf
 */


#include "gymtest/env/lunarenv.h"
#include "gymtest/env/airenv.h"
#include "gymtest/lunarnets/lunarfcnet.h"
#include "gymtest/airnets/aircnnnet.h"
#include "gymtest/env/envutils.h"

#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <iostream>
#include <fstream>
#include <algorithm>

namespace {
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("testrawdqn"));

struct RawPolicy {
	float epsilon;
	int frameNum;
	const int actionNum;
	const int decayFrameNum;

	RawPolicy(float e, int num): epsilon(e), actionNum(num), decayFrameNum(256) {
		frameNum = 0;
	}
	~RawPolicy() {};

	void getActions(std::vector<long>& actions) {
		LOG4CXX_DEBUG(logger, "getActions");
		auto randTensor = torch::rand({(long)actions.size()});
		std::vector<float> randValue(randTensor.data_ptr<float>(), randTensor.data_ptr<float>() + randTensor.numel());
		LOG4CXX_DEBUG(logger,"get randValue: " << randValue);
		auto randActionTensor = torch::randint(0, actionNum, randTensor.sizes());
//		LOG4CXX_DEBUG(logger, "ranActionTensor: " << randActionTensor);
		std::vector<float> randAction(randActionTensor.data_ptr<float>(), randActionTensor.data_ptr<float>() + randActionTensor.numel());
		LOG4CXX_DEBUG(logger, "get randActions: " << randAction);
		for (int i = 0; i < actions.size(); i ++) {
			if (randValue[i] < epsilon) {
				actions[i] = randAction[i];
			}
		}
		LOG4CXX_DEBUG(logger, "End of getActions");

		frameNum ++;
		if (frameNum >= decayFrameNum) {
			frameNum = 0;
			epsilon = std::max(epsilon / 2, (float)0.1);
		}
	}
};

bool allDone(const std::vector<bool>& dones) {
	for (const auto& done: dones) {
		if (!done) {
			return false;
		}
	}
	return true;
}

bool oneDone(const std::vector<bool>& dones) {
	for (const auto& done: dones) {
		if (done) {
			return true;
		}
	}
	return false;
}

void getRewardSum(const std::vector<float>& rewards, const std::vector<bool>& doneMark, std::vector<float>& rewardSum) {
	for (int i = 0; i < doneMark.size(); i ++) {
		if (!doneMark[i]) {
			rewardSum[i] += rewards[i];
		}
	}
}

float maxReward(const std::vector<float>& rewards) {
	float maxReward = rewards[0];
	for (const auto& reward: rewards) {
		if (reward > maxReward) {
			maxReward = reward;
		}
	}

	return maxReward;
}

float aveReward(const std::vector<float>& rewards) {
	float rewardSum = 0;
	for (const auto& reward: rewards) {
		rewardSum += reward;
	}

	return (rewardSum / rewards.size());
}

void rawTrain(const int clientNum, const int epNum) {
//	const int clientNum = num;
	torch::Device deviceType = torch::kCUDA;
	RawPolicy policy(0.1, 18);
	float gamma = 0.99;
//	const int epNum = 32;

	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, "Alien-v0", clientNum);
	auto lossComputer = torch::nn::SmoothL1Loss();

	auto info = env.init();
	auto actionSpace = std::get<1>(info);
	auto obSpace = std::get<0>(info);

    LOG4CXX_DEBUG(logger, "Action space: " << actionSpace.type << ", " << actionSpace.shape);
    LOG4CXX_DEBUG(logger, "Observation space:" << obSpace.type << "-" << obSpace.shape);

    AirCnnNet model(18);
    model.to(deviceType);
//    torch::optim::RMSprop optimizer(model.parameters());
//	torch::optim::RMSprop optimizer(net.parameters(), torch::optim::RMSpropOptions(1e-3).eps(1e-8).alpha(0.99));
	torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3));
//
//	torch::optim::SGD optimizer (
//			model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));


    std::vector<float> maxRewards;
    std::vector<float> aveRewards;
    std::vector<int> maxLens;
    for (int epCount = 0; epCount < epNum; epCount ++) {
    	LOG4CXX_INFO(logger, "episode " << epCount);
    	int epLen = 0;
    	std::vector<float> rewardSum(clientNum, 0);
    bool isDone = false;
    auto stateVec = env.reset();

    torch::Tensor doneCpuMask = torch::ones({clientNum, 1}, torch::kCPU);
    torch::Tensor doneGpuMask = torch::ones({clientNum, 1}, deviceType);
//    float* doneMaskData = doneCpuMask.data_ptr<float>();
    torch::Tensor lastRewards = torch::zeros({clientNum, 1}, deviceType);
    torch::Tensor lastStates = torch::zeros({clientNum, 4, 84, 84}, deviceType);
    torch::Tensor inputMask = torch::ones({clientNum, 4, 84, 84}, deviceType);
    torch::Tensor lastInputMask = torch::zeros({clientNum, 4, 84, 84}, deviceType);
    std::vector<bool> doneMark(clientNum, false);
    while (!isDone) {
    	epLen ++;
    	torch::Tensor input = torch::from_blob(stateVec.data(), {clientNum, 4, 84, 84}).to(deviceType);
    	input = input * inputMask + lastStates * lastInputMask;

    	torch::Tensor outputTensor = model.forward(input);
    	LOG4CXX_DEBUG(logger, "outputTensor: " << outputTensor);
    	auto actionTensor = outputTensor.argmax(-1).to(torch::kCPU);
    	LOG4CXX_DEBUG(logger, "actionTensor: " << actionTensor);
    	std::vector<long> actions(actionTensor.data_ptr<long>(), actionTensor.data_ptr<long>() + actionTensor.numel());
    	policy.getActions(actions);
    	LOG4CXX_DEBUG(logger, "Input actions; " << actions);

    	LOG4CXX_DEBUG(logger, "step");
    	auto result = env.step(actions);
    	auto nextStateVec = std::get<0>(result);
    	auto rewardVec = std::get<1>(result);
    	LOG4CXX_DEBUG(logger, "Rewards: " << rewardVec);
    	auto doneVec = std::get<2>(result);
    	getRewardSum(rewardVec, doneMark, rewardSum);

    	for (int i = 0; i < doneVec.size(); i ++) {
    		if (doneVec[i] && (!doneMark[i])) {
    			doneMark[i] = true;
//    			doneMaskData[i] = 0;
    			doneGpuMask[i][0] = 0;
    			lastRewards[i][0] = rewardVec[i];
    			lastStates[i] = input[i];
    			inputMask[i] = torch::zeros({4, 84, 84}, deviceType);
    			lastInputMask[i] = torch::ones({4, 84, 84}, deviceType);
    		}
    	}
    	isDone = oneDone(doneMark);



    	//TODO: If update of actionVec mapped on actionTensor
    	torch::Tensor indexTensor = actionTensor.to(deviceType).unsqueeze(-1);
    	LOG4CXX_DEBUG(logger, "index: " << indexTensor);
    	LOG4CXX_DEBUG(logger, "outputTensor: " << outputTensor.sizes());
    	torch::Tensor lossInput = outputTensor.gather(-1, indexTensor);
    	LOG4CXX_DEBUG(logger, "loss Input " << lossInput.sizes());
    	torch::Tensor nextStateInput = torch::from_blob(nextStateVec.data(), {clientNum, 4, 84, 84}).to(deviceType);
    	torch::Tensor nextOutput = model.forward(nextStateInput).detach();
    	LOG4CXX_DEBUG(logger, "Get next output");
    	torch::Tensor rewardTensor = torch::from_blob(rewardVec.data(), {clientNum, 1}).to(deviceType);
    	LOG4CXX_DEBUG(logger, "Get reward tensor: " << rewardTensor.sizes());
    	torch::Tensor target = std::get<0>(nextOutput.max(-1)).unsqueeze(-1) * gamma;
    	LOG4CXX_DEBUG(logger, "target: " << target.sizes());
    	target =  target + rewardTensor;
    	target = target * doneGpuMask + lastRewards;
    	LOG4CXX_DEBUG(logger, "Get target: " << target);
    	auto loss = lossComputer->forward(lossInput, target);
    	float lossValue = loss.to(torch::kCPU).item().to<float>();
    	LOG4CXX_DEBUG(logger, "Get loss: " << lossValue);
    	optimizer.zero_grad();
    	loss.backward();
    	optimizer.step();

    	stateVec = nextStateVec;
    }
    LOG4CXX_INFO(logger, "Max len: " << epLen);
    LOG4CXX_INFO(logger, "Max reward: " << maxReward(rewardSum));
    LOG4CXX_INFO(logger, "Ave reward: " << aveReward(rewardSum));
    maxRewards.push_back(maxReward(rewardSum));
    aveRewards.push_back(aveReward(rewardSum));
    maxLens.push_back(epLen);
    }

    LOG4CXX_INFO(logger, "rewards: " << maxRewards);
    LOG4CXX_INFO(logger, "lens: " << maxLens);

    std::ofstream statFile;
    statFile.open("./stats.txt");
    statFile << "rewards: " << std::endl;
    for (const auto& reward: maxRewards) {
    	statFile << reward << ", ";
    }
    statFile << std::endl;
    for (const auto& reward: aveRewards) {
    	statFile << reward << ", ";
    }
    statFile << std::endl;
    for (const auto& len: maxLens) {
    	statFile << len << ", ";
    }
    statFile << std::endl;
    statFile.close();
}

struct Stat {
	std::vector<float> rewards;
	std::vector<float> aveRewards;
	std::vector<float> lens;
	std::vector<float> aveLens;
	float aveLen;
	float aveReward;
	int epCount;
	const std::string fileName;
	std::ofstream statFile;

	Stat(std::string fName): aveLen(0), aveReward(0), epCount(0), fileName(fName) {
		statFile.open(fileName);
	}

	~Stat() {
		statFile.close();
	}

	void update(float len, float reward) {
		rewards.push_back(reward);
		lens.push_back(len);

		aveLen = (aveLen * epCount + len) / (epCount + 1);
		aveReward = (aveReward * epCount + reward) / (epCount + 1);

		aveLens.push_back(aveLen);
		aveRewards.push_back(aveReward);

		epCount ++;

		statFile << epCount << ", " << reward << ", " << len << ", " << aveReward << ", " << aveLen << std::endl;
	}

	void printCurStat() {
		if (epCount > 0) {
			LOG4CXX_INFO(logger, "ep" << epCount << ": " << rewards[epCount - 1] << ", " << lens[epCount - 1] << ", " << aveLen << ", " << aveReward);
		}
	}

	void saveVec(std::ofstream& file, const std::vector<float>& datas, const std::string comment) {
		file << comment << ", ";
		for (const auto& data: datas) {
			file << data << ", ";
		}
		file << std::endl;
	}

//	void saveTo(std::string fileName) {
//		std::ofstream statFile;
//		statFile.open("./stats.txt");
//		for (int i = 0; i < rewards.size(); i ++) {
//			statFile << i << ", " << rewards[i] << ", " << lens[i] << ", " << aveRewards[i] << ", " << aveLens[i] << std::endl;
//		}
//		statFile.close();
//	}
};

void updateReward(const std::vector<float>& rewards, std::vector<float>& rewardStat) {
	for (int i = 0; i < rewards.size(); i ++) {
		rewardStat[i] += rewards[i];
	}
}

void updateLens(std::vector<float>& lenStat) {
	for (int i = 0; i < lenStat.size(); i ++) {
		lenStat[i] ++;
	}
}

void rawResetTrain(const int clientNum, const int epNum) {
//	const int clientNum = num;
	torch::Device deviceType = torch::kCUDA;
	RawPolicy policy(0.1, 18);
	float gamma = 0.99;
	Stat stater("./stat.txt");

	const size_t offset = 4 * 84 * 84;
//	const int epNum = 32;

	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_DEBUG(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, "Alien-v0", clientNum);
	auto lossComputer = torch::nn::SmoothL1Loss();

	auto info = env.init();
	auto actionSpace = std::get<1>(info);
	auto obSpace = std::get<0>(info);

    LOG4CXX_INFO(logger, "Action space: " << actionSpace.type << ", " << actionSpace.shape);
    LOG4CXX_INFO(logger, "Observation space:" << obSpace.type << "-" << obSpace.shape);

//	auto stateVec = env.reset();

    AirCnnNet model(18);
    model.to(deviceType);
    //    torch::optim::RMSprop optimizer(model.parameters());
    //	torch::optim::RMSprop optimizer(net.parameters(), torch::optim::RMSpropOptions(1e-3).eps(1e-8).alpha(0.99));
    torch::optim::Adagrad optimizer(model.parameters(), torch::optim::AdagradOptions(1e-3));
    //
    //	torch::optim::SGD optimizer (
    //			model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));


    std::vector<float> rewardStat(clientNum, 0);
    std::vector<float> lenStat(clientNum, 0);
    int epCount = 0;
//    std::vector<std::vector<float>> resetVecs(clientNum);
//    for (int i = 0; i < clientNum; i ++) {
//    	resetVecs[i] = env.reset(i);
//    	LOG4CXX_INFO(logger, "first reset " << i << ": " << resetVecs[i].size());
//    }
//    std::vector<float> stateVec = EnvUtils::FlattenVector(resetVecs);
	auto stateVec = env.reset();
    while (epCount < epNum) {
//    	LOG4CXX_INFO(logger, "step: " << stateVec.size());to
    	torch::Tensor input = torch::from_blob(stateVec.data(), {clientNum, 4, 84, 84}).to(deviceType);
//    	LOG4CXX_INFO(logger, "cpu tensor");
//    	input = input.to(deviceType);

    	torch::Tensor outputTensor = model.forward(input);
    	LOG4CXX_DEBUG(logger, "outputTensor: " << outputTensor);
    	auto actionTensor = outputTensor.argmax(-1).to(torch::kCPU);
    	LOG4CXX_DEBUG(logger, "actionTensor: " << actionTensor);
    	std::vector<long> actions(actionTensor.data_ptr<long>(), actionTensor.data_ptr<long>() + actionTensor.numel());
    	policy.getActions(actions);
    	LOG4CXX_DEBUG(logger, "Input actions; " << actions);

    	LOG4CXX_DEBUG(logger, "step");
    	auto result = env.step(actions);
    	auto nextStateVec = std::get<0>(result);
    	auto rewardVec = std::get<1>(result);
    	LOG4CXX_DEBUG(logger, "Rewards: " << rewardVec);
    	auto doneVec = std::get<2>(result);
    	auto targetMask = torch::ones({clientNum, 1}).to(deviceType);

    	updateReward(rewardVec, rewardStat);
    	updateLens(lenStat);
    	for (int i = 0; i < doneVec.size(); i ++) {
    		if (doneVec[i]) {
    			epCount ++;

    			auto resetResult = env.reset(i);
    			//udpate nextstatevec, target mask
    			std::copy(resetResult.begin(), resetResult.end(), nextStateVec.begin() + (offset * i));
    			targetMask[i][0] = 0;

    			stater.update(lenStat[i], rewardStat[i]);
    			lenStat[i] = 0;
    			rewardStat[i] = 0;

    			stater.printCurStat();
    		}
    	}



    	torch::Tensor indexTensor = actionTensor.to(deviceType).unsqueeze(-1);
    	LOG4CXX_DEBUG(logger, "index: " << indexTensor);
    	LOG4CXX_DEBUG(logger, "outputTensor: " << outputTensor.sizes());
    	torch::Tensor lossInput = outputTensor.gather(-1, indexTensor);
    	LOG4CXX_DEBUG(logger, "loss Input " << lossInput.sizes());
    	torch::Tensor nextStateInput = torch::from_blob(nextStateVec.data(), {clientNum, 4, 84, 84}).to(deviceType);
    	torch::Tensor nextOutput = model.forward(nextStateInput).detach();
    	LOG4CXX_DEBUG(logger, "Get next output");
    	torch::Tensor rewardTensor = torch::from_blob(rewardVec.data(), {clientNum, 1}).to(deviceType);
    	LOG4CXX_DEBUG(logger, "Get reward tensor: " << rewardTensor.sizes());
    	torch::Tensor target = std::get<0>(nextOutput.max(-1)).unsqueeze(-1) * gamma;
    	LOG4CXX_DEBUG(logger, "target: " << target.sizes());
    	target = target * targetMask + rewardTensor;
    	LOG4CXX_DEBUG(logger, "Get target: " << target);
    	auto loss = lossComputer->forward(lossInput, target);
    	float lossValue = loss.to(torch::kCPU).item().to<float>();
    	LOG4CXX_DEBUG(logger, "Get loss: " << lossValue);
    	optimizer.zero_grad();
    	loss.backward();
    	torch::nn::utils::clip_grad_value_(model.parameters(), 1);
    	optimizer.step();

    	stateVec = nextStateVec;
    }
    stater.printCurStat();

//    stater.saveTo("./stats.txt");
}

void testReset() {
	const int clientNum = 2;
	std::string serverAddr = "tcp://127.0.0.1:10201";
	LOG4CXX_INFO(logger, "To connect to " << serverAddr);
	AirEnv env(serverAddr, "Alien-v0", clientNum);

	auto info = env.init();
	auto actionSpace = std::get<1>(info);
	auto obSpace = std::get<0>(info);

    LOG4CXX_INFO(logger, "Action space: " << actionSpace.type << ", " << actionSpace.shape);
    LOG4CXX_INFO(logger, "Observation space:" << obSpace.type << "-" << obSpace.shape);

    bool isDone = false;
    auto obsv = env.reset();
    LOG4CXX_INFO(logger, "reset " << obsv.size());
    while (!isDone) {
    	auto actions = std::vector<long>(clientNum, 3);
    	auto stepResult = env.step(actions);
    	obsv = std::get<0>(stepResult);
    	auto rewardVec = std::get<1>(stepResult);
    	auto doneVec = std::get<2>(stepResult);
//    	LOG4CXX_INFO(logger, "reward: " << rewardVec);

    	for (int i = 0; i < doneVec.size(); i ++) {
    		if (doneVec[i]) {
    			auto tmpObs = env.reset(i);
    			LOG4CXX_INFO(logger, "Reset client " << i << "result: " << tmpObs.size());
    		}
    	}
    }
}

}

int main(int argc, char** argv) {
	log4cxx::BasicConfigurator::configure();
//	log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getDebug());
	log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getInfo());

//	rawTrain(atoi(argv[1]), atoi(argv[2]));
	rawResetTrain(atoi(argv[1]), atoi(argv[2]));
//	testReset();

	LOG4CXX_INFO(logger, "End of test");
}

