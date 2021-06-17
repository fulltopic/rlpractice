/*
 * lunarenv.cpp
 *
 *  Created on: Apr 2, 2021
 *      Author: zf
 */


#include "gymtest/env/lunarenv.h"
#include "gymtest/env/envutils.h"

//const std::string LunarEnv::EnvName = "LunarLander-v2";
//const std::string LunarEnv::EnvName = "AirRaid-v0";


LunarEnv::LunarEnv(std::string url, std::string envName, int instNum) :
		clientNum(instNum), EnvName(envName),
	logger(log4cxx::Logger::getLogger("LunarEnv_" + url)),
	comm(url)
{

}

//const int envNum = 2;
std::pair<cpprl::ActionSpace, cpprl::ActionSpace>
	LunarEnv::init() {
	LOG4CXX_INFO(logger, "To make env for " << EnvName);
	auto makeParam = std::make_shared<MakeParam>();
	makeParam->env_name = EnvName;
	makeParam->num_envs = clientNum;
	Request<MakeParam> makeReq("make", makeParam);
	comm.send_request(makeReq);
    LOG4CXX_INFO(logger, "Make response: " << comm.get_response<MakeResponse>()->result);

    Request<InfoParam> infoReq("info", std::make_shared<InfoParam>());
    comm.send_request(infoReq);
    auto envInfo = comm.get_response<InfoResponse>();
    LOG4CXX_INFO(logger, "Action space: " << envInfo->action_space_type << ", " << envInfo->action_space_shape);
    LOG4CXX_INFO(logger, "Observation space:" << envInfo->observation_space_type << "-" << envInfo->observation_space_shape);

    auto obsvShape = envInfo->observation_space_shape;
    obsvShape.insert(obsvShape.begin(), clientNum);

    cpprl::ActionSpace actionSpace{envInfo->action_space_type, envInfo->action_space_shape};
    cpprl::ActionSpace obsvSpace{envInfo->observation_space_type, obsvShape};
    return std::make_pair(obsvSpace, actionSpace);
}

std::vector<float> LunarEnv::reset() {
	LOG4CXX_DEBUG(logger, "To reset env");

    auto resetParam = std::make_shared<ResetParam>();
    resetParam->x = -1;
    Request<ResetParam> resetReq("reset", resetParam);
    comm.send_request(resetReq);
    LOG4CXX_DEBUG(logger, "Reseted");

    std::vector<float> obsvVec = EnvUtils::FlattenVector(comm.get_response<MlpResetResponse>()->observation);

    return obsvVec;
}

std::vector<float> LunarEnv::reset(const int id) {
	LOG4CXX_DEBUG(logger, "To reset env: " << id);

    auto resetParam = std::make_shared<ResetParam>();
    resetParam->x = id;
    Request<ResetParam> resetReq("reset", resetParam);
    comm.send_request(resetReq);
    LOG4CXX_DEBUG(logger, "reset request sent");

//    std::vector<float> obsvVec = EnvUtils::FlattenVector(comm.get_response<CnnResetResponse>()->observation);
    auto rsp = comm.get_response<MlpResetResponse>()->observation;
    std::vector<float> obsvVec = EnvUtils::FlattenVector(rsp);
    LOG4CXX_DEBUG(logger, "reset");

    //tmp
//    updateTmpObs(id, obsvVec);
//    int otherId = 1 - id;
//    int diff = cmpTmpObs(otherId, EnvUtils::FlattenVector(rsp[otherId]));
//    LOG4CXX_INFO(logger, "Untouched " << diff);

    return obsvVec;
}

std::tuple<std::vector<float>, float, bool>
		LunarEnv::step(const int action, const bool render) {
	auto stepParam = std::make_shared<StepParam>();
	std::vector<std::vector<float>> actions(clientNum, std::vector<float>(1, action));
	stepParam->actions = actions;
	stepParam->render = render;
	Request<StepParam> stepReq("step", stepParam);
	comm.send_request(stepReq);

	auto stepResult = comm.get_response<MlpStepResponse>();
	auto obsvVec = EnvUtils::FlattenVector(stepResult->observation);

	auto rawRewardVec = EnvUtils::FlattenVector(stepResult->real_reward);
	float reward = rawRewardVec[0];


    auto rawDone = stepResult->done;
    bool done = rawDone[0][0];

    return {std::move(obsvVec), reward, done};
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<bool>>
	LunarEnv::step(const std::vector<long> actions, const bool render) {
	auto stepParam = std::make_shared<StepParam>();
	std::vector<std::vector<float>> actionParam(actions.size(), std::vector<float>(1, 0));
	for (int i = 0; i < actions.size(); i ++) {
		actionParam[i][0] = actions[i];
	}
	stepParam->actions = actionParam;
	stepParam->render = render;
	Request<StepParam> stepReq("step", stepParam);
	comm.send_request(stepReq);

	auto stepResult = comm.get_response<MlpStepResponse>();
	auto obsvVec = EnvUtils::FlattenVector(stepResult->observation);
	auto rewardVec = EnvUtils::FlattenVector(stepResult->real_reward);
	auto doneVec = EnvUtils::FlattenVector(stepResult->done);

	return {obsvVec, rewardVec, doneVec};
}

