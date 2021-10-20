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

	epsilonInput = register_buffer("epsilonInput", torch::zeros({1, inFeatures}));
	epsilonOutput = register_buffer("epsilonOutput", torch::zeros({outFeatures, 1}));

	resetParams();
	resetNoise();

	std::cout << "-----------------------------------------------> reset" << std::endl;
}

void NoisyLinear::resetParams() {
	float stdValue = (float)sqrt(3 / (float)inFeatures);

	torch::nn::init::uniform_(wMu, -stdValue, stdValue);
	torch::nn::init::uniform_(bMu, -stdValue, stdValue); //bias always defined

//	float wSigmaValue = stdInit / (float)sqrt(wSigma.size(1));
//	float bSigmaValue = stdInit / (float)sqrt(bSigma.size(0));
	float wSigmaValue = stdInit / (float)sqrt(inFeatures);
	float bSigmaValue = stdInit / (float)sqrt(outFeatures);

	torch::nn::init::constant_(wSigma, 0.017);
	torch::nn::init::constant_(bSigma, 0.017);

//	std::cout << "resetParams" << std::endl;
//	std::cout << "wMu: " << wMu << std::endl;
//	std::cout << "bMu: " << bMu << std::endl;
//	std::cout << "wSigma: " << wSigma << std::endl;
//	std::cout << "bSigma: " << bSigma << std::endl;
}

void NoisyLinear::resetNoise() {
	wEpsilon.normal_();
	bEpsilon.normal_();

//	epsilonInput.normal_();
//	epsilonOutput.normal_();
//
//	auto input = (epsilonInput.sign() * epsilonInput.abs().sqrt()).to(epsilonInput.device());
//	auto output = epsilonOutput.sign() * epsilonOutput.abs().sqrt().to(epsilonOutput.device());
//
//	wEpsilon = input * output;
//	bEpsilon.copy_(output.squeeze(1));

//	std::cout << "w ep: " << wEpsilon << std::endl;
//	std::cout << "b ep: " << bEpsilon << std::endl;
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

//		std::cout << "wMu " << wMu << std::endl;
//		std::cout << "bMu " << bMu << std::endl;
//		std::cout << "wSigma " << wSigma << std::endl;
//		std::cout << "bSigma " << bSigma << std::endl;
//		std::cout << "wSigma " << wSigma.sizes() << std::endl;
//		std::cout << "wEpsilon " << wEpsilon.sizes() << std::endl;
//		std::cout << "mul " << wSigma.mul(wEpsilon).sizes() << std::endl;
//		std::cout << "element " << (wSigma * wEpsilon).sizes() << std::endl;
	} else {
//		std::cout << "------------------------------------> eval " << std::endl;
		weight = wMu;
		bias = bMu;
	}

//	weight = wMu + wSigma * wEpsilon;
//	bias = bMu;

//	weight = wMu + wSigma * wEpsilon;
//	bias = bMu + bSigma * bEpsilon;
	torch::Tensor output = torch::nn::functional::linear(input, weight, bias);
	return output;
}


