/*
 * a3cclientdata.cpp
 *
 *  Created on: Nov 7, 2021
 *      Author: zf
 */


#include "a3cclientData.h"



A3CClientData::A3CClientData(std::vector<at::IntArrayRef> shapes) {
	for (int i = 0; i < shapes.size(); i ++) {
		grads.push_back(torch::zeros(shapes[i]));
		targetParams.push_back(torch::zeros(shapes[i]));
	}
}

void A3CClientData::resetGrad() {
	for (int i = 0; i < grads.size(); i ++) {
		grads[i].fill_(0);
	}
}

void A3CClientData::resetTargetParam() {
	for (int i = 0; i < targetParams.size(); i ++) {
		targetParams[i].fill_(0);
	}
}

void A3CClientData::updateGrad(std::vector<torch::Tensor> deltaGrads) {
	torch::NoGradGuard guard;

	for (int i = 0; i < grads.size(); i ++) {
		grads[i].add_(deltaGrads[i]);
	}
}

void A3CClientData::syncTarget(std::vector<torch::Tensor> params) {
	torch::NoGradGuard guard;

	auto startIndex = deltaGrads[1].item<long>();
	auto endIndex = deltaGrads[1].item<long>();
	for (int i = 0; i < targetParams.size(); i ++) {
		targetParams[i].copy_(params[i]);
	}
}

const std::vector<torch::Tensor>& A3CClientData::getGrad() {
	return grads;
}
