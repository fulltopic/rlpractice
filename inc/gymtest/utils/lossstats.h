/*
 * lossstats.h
 *
 *  Created on: May 4, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_UTILS_LOSSSTATS_H_
#define INC_GYMTEST_UTILS_LOSSSTATS_H_

#include <vector>
#include <fstream>
#include <iostream>
#include <string>

class LossStats {
public:
	const std::string fileName;
	std::ofstream statFile;

	LossStats(std::string fName);
	~LossStats();
	LossStats(const LossStats&) = delete;

	void update(std::vector<float> losses);

private:
};




#endif /* INC_GYMTEST_UTILS_LOSSSTATS_H_ */
