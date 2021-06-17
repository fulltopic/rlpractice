/*
 * airenv.cpp
 *
 *  Created on: Apr 5, 2021
 *      Author: zf
 */


#include "gymtest/env/airenv.h"
#include "gymtest/env/envutils.h"

//const std::string LunarEnv::EnvName = "LunarLander-v2";
//const std::string AirEnv::EnvName = "AirRaid-v0";
//const std::string AirEnv::EnvName = "Alien-v0";
//const std::string AirEnv::EnvName = "SpaceInvaders-v0";


AirEnv::AirEnv(std::string url, std::string envName, int instNum) :
		clientNum(instNum), EnvName(envName),
	logger(log4cxx::Logger::getLogger("AirEnv_" + url)),
	comm(url)
{

}

//const int envNum = 2;
std::pair<cpprl::ActionSpace, cpprl::ActionSpace>
AirEnv::init() {
	LOG4CXX_DEBUG(logger, "To make env for " << EnvName);
	auto makeParam = std::make_shared<MakeParam>();
	makeParam->env_name = EnvName;
	makeParam->num_envs = clientNum;
	Request<MakeParam> makeReq("make", makeParam);
	comm.send_request(makeReq);
	auto makeRsp = comm.get_response<MakeResponse>()->result;
    LOG4CXX_INFO(logger, "Make response: " << makeRsp);

    Request<InfoParam> infoReq("info", std::make_shared<InfoParam>());
    comm.send_request(infoReq);
    auto envInfo = comm.get_response<InfoResponse>();


    auto obsvShape = envInfo->observation_space_shape;
    obsvShape.insert(obsvShape.begin(), clientNum);

    cpprl::ActionSpace actionSpace{envInfo->action_space_type, envInfo->action_space_shape};
    cpprl::ActionSpace obsvSpace{envInfo->observation_space_type, obsvShape};
    LOG4CXX_INFO(logger, "inited: " << clientNum);
    return std::make_pair(obsvSpace, actionSpace);
}


//std::vector<float> tmpObs(2 * 4 * 84 *84, 0);
//void updateTmpObs(int id, const std::vector<float>& data) {
//	int offset = id * 4 * 84 * 84;
//	for (int i = 0; i <  + 4 * 84 * 84; i ++) {
//		tmpObs[offset + i] = data[i];
//	}
//}
//
//void updateTmpObs(const std::vector<float>& data) {
//	int offset = 0;
//	for (int i = 0; i < data.size(); i ++) {
//		tmpObs[i] = data[i];
//	}
//}
//
//int cmpTmpObs(int id, const std::vector<float>& data) {
//	int offset = id * 4 * 84 * 84;
//	int diff = 0;
//	for (int i = 0; i <  + 4 * 84 * 84; i ++) {
//		if (tmpObs[offset + i] != data[i]) {
//			diff ++;
//		}
//	}
//
//	return diff;
//}

std::vector<float> AirEnv::reset() {
	LOG4CXX_DEBUG(logger, "To reset env");

    auto resetParam = std::make_shared<ResetParam>();
    resetParam->x = -1;
    Request<ResetParam> resetReq("reset", resetParam);
    comm.send_request(resetReq);
    LOG4CXX_DEBUG(logger, "reset request sent");

    std::vector<float> obsvVec = EnvUtils::FlattenVector(comm.get_response<CnnResetResponse>()->observation);
    LOG4CXX_DEBUG(logger, "reset obs: " << obsvVec.size());

    //tmp
//    updateTmpObs(obsvVec);

    return obsvVec;
}

std::vector<float> AirEnv::reset(const int id) {
	LOG4CXX_DEBUG(logger, "To reset env: " << id);

    auto resetParam = std::make_shared<ResetParam>();
    resetParam->x = id;
    Request<ResetParam> resetReq("reset", resetParam);
    comm.send_request(resetReq);
    LOG4CXX_DEBUG(logger, "reset request sent");

//    std::vector<float> obsvVec = EnvUtils::FlattenVector(comm.get_response<CnnResetResponse>()->observation);
    auto rsp = comm.get_response<CnnResetResponse>()->observation;
    std::vector<float> obsvVec = EnvUtils::FlattenVector(rsp[id]);
    LOG4CXX_DEBUG(logger, "reset " << rsp.size());

    //tmp
//    updateTmpObs(id, obsvVec);
//    int otherId = 1 - id;
//    int diff = cmpTmpObs(otherId, EnvUtils::FlattenVector(rsp[otherId]));
//    LOG4CXX_INFO(logger, "Untouched " << diff);

    return obsvVec;
}

std::tuple<std::vector<float>, float, bool>
AirEnv::step(const int action, const bool render) {
	auto stepParam = std::make_shared<StepParam>();
	std::vector<std::vector<float>> actions(clientNum, std::vector<float>(1, action));
	stepParam->actions = actions;
	stepParam->render = render;
	Request<StepParam> stepReq("step", stepParam);
	comm.send_request(stepReq);

	auto stepResult = comm.get_response<CnnStepResponse>();
	auto obsvVec = EnvUtils::FlattenVector(stepResult->observation);

	auto rawRewardVec = EnvUtils::FlattenVector(stepResult->real_reward);
	float reward = rawRewardVec[0];


    auto rawDone = stepResult->done;
    bool done = rawDone[0][0];

    return {std::move(obsvVec), reward, done};
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<bool>>
AirEnv::step(const std::vector<long> actions, const bool render) {
	auto stepParam = std::make_shared<StepParam>();
	std::vector<std::vector<float>> actionParam(actions.size(), std::vector<float>(1, 0));
	for (int i = 0; i < actions.size(); i ++) {
		actionParam[i][0] = actions[i];
	}
	stepParam->actions = actionParam;
	stepParam->render = render;
	Request<StepParam> stepReq("step", stepParam);
	comm.send_request(stepReq);

	auto stepResult = comm.get_response<CnnStepResponse>();
	auto obsvVec = EnvUtils::FlattenVector(stepResult->observation);
	auto rewardVec = EnvUtils::FlattenVector(stepResult->real_reward);
//	auto rewardVec = EnvUtils::FlattenVector(stepResult->reward);
	auto doneVec = EnvUtils::FlattenVector(stepResult->done);

//	auto sum = obsvVec[0];
//	for (const auto& ob: obsvVec) {
//		sum += ob;
//	}
//	LOG4CXX_INFO(logger, "obsvec: " << sum);

//	auto alReward = EnvUtils::FlattenVector(stepResult->reward);
//
//	bool match = true;
//	for (int i = 0; i < alReward.size(); i ++) {
//		if (rewardVec[i] != alReward[i]) {
//			match = false;
//			LOG4CXX_INFO(logger, "real reward: " << rewardVec);
//			LOG4CXX_INFO(logger, "reward: " << alReward);
//		}
//	}
//	if (match) {
//		LOG4CXX_INFO(logger, "all match");
//	}

	return {obsvVec, rewardVec, doneVec};
}

