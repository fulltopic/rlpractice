/*
 * a3cclienthandle.cpp
 *
 *  Created on: Nov 7, 2021
 *      Author: zf
 */


#include "a3c/a3cclient.h"
#include "a3c/a3cclienthandle.h"


A3CClientHandler::A3CClientHandler(std::vector<at::IntArrayRef> shapes) {
	for (int i = 0; i < shapes.size(); i ++) {
		grads.push_back(torch::zeros(shapes[i]));
		targetParams.push_back(torch::zeros(shapes[i]));
	}

	checkGradStep();
}

A3CClientHandler::~A3CClientHandler() {}

void A3CClientHandler::checkGradStep() {
	torch::Tensor cmdTensor = torch::zeros({A3CConfig::CmdLen}, longOpt);

	int index = 0;
	while (index < grads.size()) {
		int bufSize = 0;
		int curStep = 0;
		int lastStep = curStep;

		while (bufSize < A3CConfig::BufCap) {
			lastStep = curStep;
			curStep ++;
			if (curStep + index > grads.size()) {
				break;
			}

			std::vector<torch::Tensor> ts;
			ts.push_back(cmdTensor); //cmd
			for (int i = 0; i < curStep; i ++) {
				LOG4CXX_DEBUG(logger, "try push " << (i + index));
				ts.push_back(torch::zeros(grads[i + index].sizes()));
			}
			std::ostringstream os;
			torch::save(ts, os);
			bufSize = os.str().length();
			LOG4CXX_DEBUG(logger, "try index = " << index << ", curStep = " << curStep << ", bufSize = " << bufSize);


		}
		if (lastStep == 0) {
			LOG4CXX_ERROR(logger, "Single layer beyond capacity " << index << ": " << bufSize << " > " << A3CConfig::BufCap);
			break;
		}

		LOG4CXX_INFO(logger, "index = " << index << ", step = " << lastStep);
		gradSteps.push_back(lastStep);
		index = index + lastStep;
	}
}

void A3CClientHandler::updateGrad(std::vector<torch::Tensor> deltaGrads) {
	torch::NoGradGuard guard;

	for (int i = 0; i < grads.size(); i ++) {
		grads[i].add_(deltaGrads[i]);
	}
}

void A3CClientHandler::sendGrads() {
//	int startIndex = 0;
//	for(int i = 0; i < gradSteps.size(); i ++) {
//		std::vector<torch::Tensor> ts;
//
//		int endIndex = startIndex + gradSteps[i] - 1; //Inclusive
//		torch::Tensor cmdTensor = torch::zeros({A3CConfig::CmdLen}, longOpt);
//		cmdTensor[0] = A3CConfig::AddGrad;
//		cmdTensor[1] = startIndex;
//		cmdTensor[2] = endIndex;
//		ts.push_back(cmdTensor);
//
//		for (int index = startIndex; index <= endIndex; index ++) {
//			LOG4CXX_INFO(logger, "push grad " << index);
//			ts.push_back(grads[index]);
//		}
//		LOG4CXX_INFO(logger, "To send " << startIndex << "~" << endIndex);
//
//		startIndex = endIndex + 1;
//
//		if (!client->send(ts)) {
//			break;
//		}
//	}
	torch::Tensor cmdTensor = torch::zeros({A3CConfig::CmdLen}, longOpt);
	cmdTensor[0] = A3CConfig::AddGrad;
	cmdTensor[1] = 0;
	cmdTensor[2] = (long)(grads.size() - 1);

	std::vector<torch::Tensor> ts;
	ts.push_back(cmdTensor);
	for (int i = 0; i < grads.size(); i ++) {
		ts.push_back(grads[i]);
	}

	client->send(ts);
}

const std::vector<torch::Tensor>& A3CClientHandler::getTargets() {
	return targetParams;
}

void A3CClientHandler::setClient(std::shared_ptr<A3CClient> cp) {
	client = cp;
}

void A3CClientHandler::processRcv(std::vector<torch::Tensor>& ts) {
	long* dataPtr = ts[0].data_ptr<long>();
	int cmd = dataPtr[0];
//	A3CConfig::Cmd cmdEnum(cmd);

	switch(cmd) {
	case A3CConfig::SyncTarget:
		syncTarget(ts);
		break;
	case A3CConfig::Invalid:
		LOG4CXX_ERROR(logger, "invalid command " << cmd);
		break;
	default:
		LOG4CXX_ERROR(logger, "unexpected command " << cmd);
	}
}


void A3CClientHandler::syncTarget(std::vector<torch::Tensor> params) {
	long* dataPtr = params[0].data_ptr<long>();
	int startIndex = dataPtr[1];
	int endIndex = dataPtr[2];

	for (int i = startIndex; i <= endIndex; i ++) {
		targetParams[i].copy_(params[i - startIndex + 1]);
	}
}

