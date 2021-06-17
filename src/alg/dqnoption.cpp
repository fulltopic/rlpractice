/*
 * dqnoption.cpp
 *
 *  Created on: Apr 11, 2021
 *      Author: zf
 */



#include "alg/dqnoption.h"

DqnOption::DqnOption(at::IntArrayRef iShape, torch::Device dType, int cap, float gm, std::string path, int tUpdate):
	inputShape(iShape),
	deviceType(dType),
	rbCap(cap),
	gamma(gm),
	statPath(path),
	targetUpdate(tUpdate){
	stateSize = 1;
	for (int i = 0; i < iShape.size(); i ++) {
		stateSize *= iShape[i];
	}
}

