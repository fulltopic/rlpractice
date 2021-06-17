/*
 * inputnorm.h
 *
 *  Created on: May 14, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_UTILS_INPUTNORM_H_
#define INC_GYMTEST_UTILS_INPUTNORM_H_

#include <torch/torch.h>
#include <string>
#include <cstdint>

class InputNorm {
private:
	float count = 0;
	torch::Tensor mean;
	torch::Tensor var;

	torch::Device deviceType;


public:
	InputNorm(torch::Device dType);;
	~InputNorm() = default;
	InputNorm(const InputNorm&) = delete;

	void update(torch::Tensor input, float batchNum);
	inline float getMean() { return mean.item<float>(); }
	inline float getVar() { return var.sqrt().item<float>(); }

	void save(std::string prefix);
	void load(std::string prefix);
};



#endif /* INC_GYMTEST_UTILS_INPUTNORM_H_ */
