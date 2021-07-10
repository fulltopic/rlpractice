/*
 * stats.h
 *
 *  Created on: Apr 10, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_UTILS_STATS_H_
#define INC_GYMTEST_UTILS_STATS_H_

#include <vector>
#include <fstream>
#include <iostream>
#include <string>

class Stats {
public:
	const int statGap = 1024;
	std::vector<float> rewards;
	std::vector<float> aveRewards;
	std::vector<float> lens;
	std::vector<float> aveLens;
	float aveLen;
	float aveReward;
	int epCount;
	const std::string fileName;
	std::ofstream statFile;

	Stats(std::string fName, int gap = 1024);

	~Stats();

	void update(float len, float reward);
	void saveVec(std::ofstream& file, const std::vector<float>& datas, const std::string comment);
	inline const int getIndex() const { return index;}
	std::vector<float> getCurState();

	static void UpdateReward(std::vector<float>& target, const std::vector<float>& src, bool toClip = false, float minClip = -1, float maxClip = 1);
	static void UpdateLen(std::vector<float>& lens);


private:
	int index = 0;
};

std::ostream& operator<< (std::ostream& os, const Stats& st);


#endif /* INC_GYMTEST_UTILS_STATS_H_ */
