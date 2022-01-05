/*
 * dqnoption.cpp
 *
 *  Created on: Apr 11, 2021
 *      Author: zf
 */



#include "alg/utils/dqnoption.h"

DqnOption::DqnOption(at::IntArrayRef iShape, torch::Device dType, int cap, float gm, std::string path, int tUpdate):
	inputShape(iShape),
	testInputShape(inputShape), //TODO
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

DqnOption::DqnOption(at::IntArrayRef iShape, at::IntArrayRef tShape, torch::Device dType):
		inputShape(iShape),
		testInputShape(tShape), //TODO
		deviceType(dType)
{

}
