/*
 * noisylinear.cpp
 *
 *  Created on: Aug 31, 2021
 *      Author: zf
 */

#include "gymtest/noisynets/NoisyLinear.h"
#include <torch/torch.h>

#include <iostream>
#include <math.h>

//TODO: No util init to this layer
NoisyLinear::NoisyLinear(int in, int out, float init): inFeatures(in), outFeatures(out), stdInit(init) {
	reset();
}

void NoisyLinear::reset() {
	wMu = register_parameter("wMu", torch::empty({outFeatures, inFeatures}));
	wSigma = register_parameter("wSigma", torch::empty({outFeatures, inFeatures}));
	wEpsilon = register_buffer("wEpsilon", torch::zeros({outFeatures, inFeatures}, floatOpt));

	bMu = register_parameter("bMu", torch::empty({outFeatures}));
	bSigma = register_parameter("bSigma", torch::empty({outFeatures}));
	bEpsilon = register_buffer("bEpsilon", torch::zeros({outFeatures}, floatOpt));

	resetParams();
	resetNoise();
}

void NoisyLinear::resetParams() {
	float stdValue = (float)sqrt(3 / inFeatures);

	torch::nn::init::uniform_(wMu, -stdValue, stdValue);
	torch::nn::init::uniform_(bMu, -stdValue, stdValue); //bias always defined

	float wSigmaValue = stdInit / (float)sqrt(wSigma.size(1));
	float bSigmaValue = stdInit / (float)sqrt(bSigma.size(0));

	torch::nn::init::constant_(wSigma, wSigmaValue);
	torch::nn::init::constant_(bSigma, bSigmaValue);
}

void NoisyLinear::resetNoise() {
	wEpsilon.normal_();
	bEpsilon.normal_();
}

torch::Tensor NoisyLinear::forward(torch::Tensor input) {
//	resetNoise();

	torch::Tensor weight;
	torch::Tensor bias;
	if (this->is_training()) {
		weight = wMu + wSigma * wEpsilon;
		bias = bMu + bSigma * bEpsilon;
//		std::cout << "training " << std::endl;
//		weight = wMu + wSigma.mul(wEpsilon);
//		bias = bMu + bSigma.mul(bEpsilon);

//		std::cout << "wSigma " << wSigma.sizes() << std::endl;
//		std::cout << "wEpsilon " << wEpsilon.sizes() << std::endl;
//		std::cout << "mul " << wSigma.mul(wEpsilon).sizes() << std::endl;
//		std::cout << "element " << (wSigma * wEpsilon).sizes() << std::endl;
	} else {
//		std::cout << "------------------------------------> eval " << std::endl;
		weight = wMu;
		bias = bMu;
	}

	torch::Tensor output = torch::nn::functional::linear(input, weight, bias);
	return output;
}


