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

	if (len > maxLen) {
		maxLen = len;
	}
	if (reward > maxReward) {
		maxReward = reward;
	}

	epCount ++;

	statFile << epCount << ", " << reward << ", " << len << ", " << aveReward << ", " << aveLen << ", " << maxReward << ", " << maxLen << std::endl;
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

void Stats::UpdateReward(std::vector<float>& target, const std::vector<float>& src, bool toClip, float minClip, float maxClip) {
	for (int i = 0; i < target.size(); i ++) {
		if (toClip) {
			if (src[i] < minClip) {
				target[i] += minClip;
			} else if(src[i] > maxClip) {
				target[i] += maxClip;
			} else {
				target[i] += src[i];
			}
		} else {
			target[i] += src[i];
		}
	}
}
void Stats::UpdateLen(std::vector<float>& lens) {
	for (int i = 0; i < lens.size(); i ++) {
		lens[i] += 1;
	}
}

