/*
 * dqnoption.h
 *
 *  Created on: Apr 11, 2021
 *      Author: zf
 */

#ifndef INC_ALG_DQNOPTION_H_
#define INC_ALG_DQNOPTION_H_

#include <torch/torch.h>
#include <string>

class DqnOption {
public:
	const at::IntArrayRef inputShape;
	float gamma;
	torch::Device deviceType;
	int totalStep = 128;
	int envNum = 16;

	//input
	bool isAtari = true;
	float inputScale = 1;

	//reward
	int rewardScale = 1;
	int rewardMin = -1;
	int rewardMax = 1;
	bool normReward = false;
	bool clipRewardStat = false;

	//reward penalty
	bool toPunish = false;
	int penalStep = 400;
	float penalReward = -0.5;

	//Load/save model
	std::string statPathPrefix;
	std::string statPath;
	std::string teststatPath = "./test_stat.txt";
	bool saveModel = false;
	std::string savePathPrefix = "";
	bool loadModel = false;
	bool loadOptimizer = true;
	std::string loadPathPrefix = "";
	float saveThreshold = 1e+8;
	float saveStep = 1000;
	float sumSaveThreshold = 1e+8;
	float sumSaveStep = 1000;

	//Target update
	int targetUpdate;

	//Explore
	float exploreBegin = 1;
	float exploreEnd = 0.1;
	float exploreDecay = 0.1;
	float exploreEp = 1;
	float exploreStep = 1;
	float explorePhase = 10;
	float startStep = 1;
	int targetUpdateStep = 1024;

	//Replay buffer
	int rbCap;
	std::string dbPath = "";
	int stateSize;

	//Prioritized buffer
	float pbAlpha = 0.6;
	float pbBetaBegin = 0.4;
	float pbBetaEnd = 1;
	float pbEpsilon = 1e-6;

	//a2c
	int batchSize;
	float maxGradNormClip = 0.5;
	float entropyCoef = 1e-3;
	float valueCoef = 0.5;
	int statCap = 1024;
	//ppo
	int trajStepNum = 1;
	int epochNum = 1;
	float ppoLambda = 0.6;
	float ppoEpsilon = 0.1;
	bool klEarlyStop = false;
	float maxKl = 0.01;
	bool valueClip = false;
	float maxValueDelta = 1.0;

	//breakout
	bool multiLifes = false;
	int donePerEp = 1;

	//test
	bool toTest = false;
	int testGapEp = 100;
	int testBatch = 1;
	int testOutput = 1;
	float hangRewardTh = 1;
	int hangNumTh = 1;
	int randomStep = 1;
	bool randomHang = false;

	DqnOption(at::IntArrayRef iShape, torch::Device dType = torch::kCPU, int cap = 128, float gm = 0.99, std::string path = "./stat.txt", int tUpdate = 128);
	~DqnOption() = default;
};


#endif /* INC_ALG_DQNOPTION_H_ */
