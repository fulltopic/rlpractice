/*
 * stats.cpp
 *
 *  Created on: Apr 10, 2021
 *      Author: zf
 */
#include "gymtest/utils/stats.h"
//#include <log4cxx/logger.h>
//#include <log4cxx/basicconfigurator.h>

Stats::Stats(std::string fName, int gap): statGap(gap),
	aveLen(0), aveReward(0), epCount(0), fileName(fName),
	rewards(gap, 0), lens(gap, 0), aveRewards(gap, 0), aveLens(gap, 0)
	{
	statFile.open(fileName);
}

Stats::~Stats() {
	statFile.close();
}

void Stats::update(float len, float reward) {
	if (epCount < statGap) {
		index = epCount;

		rewards[index] = reward;
		lens[index] = len;

		aveLen += (len - aveLen) / (epCount + 1);
		aveReward += (reward - aveReward) / (epCount + 1);
		aveRewards[index] = aveReward;
		aveLens[index] = aveLen;
	} else {
		index = epCount % statGap;

		aveReward += (reward - rewards[index]) / statGap;
		aveLen += (len - lens[index]) / statGap;

		rewards[index] = reward;
		lens[index] = len;
		aveRewards[index] = aveReward;
		aveLens[index] = aveLen;
	}



//
//	rewards.push_back(reward);
//	lens.push_back(len);
//
//	if (rewards.size() <= statGap) {
//		aveLen += (len - aveLen) / (epCount + 1);
//		aveReward += (reward - aveReward) / (epCount + 1);
//	} else {
//		aveLen += (len - lens[lens.size() - statGap]) / statGap;
//		aveReward += (reward - rewards[rewards.size() - statGap]) / statGap;
//	}
//	aveLens.push_back(aveLen);
//	aveRewards.push_back(aveReward);

	epCount ++;

	statFile << epCount << ", " << reward << ", " << len << ", " << aveReward << ", " << aveLen << std::endl;
}



void Stats::saveVec(std::ofstream& file, const std::vector<float>& datas, const std::string comment) {
	file << comment << ", ";
	for (const auto& data: datas) {
		file << data << ", ";
	}
	file << std::endl;
}

std::vector<float> Stats::getCurState() {
	return {aveReward, aveLen};
}

//	void saveTo(std::string fileName) {
//		std::ofstream statFile;
//		statFile.open("./stats.txt");
//		for (int i = 0; i < rewards.size(); i ++) {
//			statFile << i << ", " << rewards[i] << ", " << lens[i] << ", " << aveRewards[i] << ", " << aveLens[i] << std::endl;
//		}
//		statFile.close();
//	}

std::ostream& operator<< (std::ostream& os, const Stats& st) {
	if (st.epCount > 0) {
		const int index = st.getIndex();
		os << "ep" << st.epCount << ": " << st.rewards[index] << ", " << st.lens[index] << ", " << st.aveLen << ", " << st.aveReward;
	} else {
		os << "No record ";
	}

	return os;
}

void Stats::UpdateReward(std::vector<float>& target, const std::vector<float>& src) {
	for (int i = 0; i < target.size(); i ++) {
		target[i] += src[i];
	}
}
void Stats::UpdateLen(std::vector<float>& lens) {
	for (int i = 0; i < lens.size(); i ++) {
		lens[i] += 1;
	}
}

