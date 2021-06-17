/*
 * lossstats.cpp
 *
 *  Created on: May 4, 2021
 *      Author: zf
 */



#include "gymtest/utils/lossstats.h"
//#include <log4cxx/logger.h>
//#include <log4cxx/basicconfigurator.h>

LossStats::LossStats(std::string fName): fileName(fName)
{
	statFile.open(fileName);
}

LossStats::~LossStats() {
	statFile.close();
}

void LossStats::update(std::vector<float> losses) {
	for (const auto& loss: losses) {
		statFile << loss << ",";
	}
	statFile << std::endl;
}
