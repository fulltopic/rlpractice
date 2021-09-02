/*
 * NoisyLinear.h
 *
 *  Created on: Aug 31, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_NOISYNETS_NOISYLINEAR_H_
#define INC_GYMTEST_NOISYNETS_NOISYLINEAR_H_

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/options/linear.h>
#include <torch/nn/pimpl.h>
#include <torch/nn/functional/linear.h>
#include <torch/types.h>

class NoisyLinear: public torch::nn::Module {
public:
	const int inFeatures;
	const int outFeatures;
	const float stdInit;

	torch::Tensor wMu;
	torch::Tensor wSigma;
	torch::Tensor wEpsilon;

	torch::Tensor bMu;
	torch::Tensor bSigma;
	torch::Tensor bEpsilon;

	const torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	const torch::TensorOptions floatOpt = torch::TensorOptions().dtype(torch::kFloat);


	explicit NoisyLinear(int in, int out, float init = 0.4);
	~NoisyLinear() = default;
	NoisyLinear(const NoisyLinear&) = delete;

	void reset();
	void resetParams();
	void resetNoise();

	torch::Tensor forward(torch::Tensor input);
};



#endif /* INC_GYMTEST_NOISYNETS_NOISYLINEAR_H_ */
