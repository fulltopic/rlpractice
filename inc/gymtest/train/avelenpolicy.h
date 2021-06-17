/*
 * avelenpolicy.h
 *
 *  Created on: Apr 10, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_TRAIN_AVELENPOLICY_H_
#define INC_GYMTEST_TRAIN_AVELENPOLICY_H_

#include <torch/torch.h>
#include <vector>

class AveLenPolicy {
private:
	float epsilon;
	const int actionNum;
	static torch::TensorOptions intOpt;

public:
	AveLenPolicy(float ep, int an);
	~AveLenPolicy() = default;

	std::vector<int64_t> getActions(torch::Tensor input);
	std::vector<int64_t> getTestActions(torch::Tensor input);
};






#endif /* INC_GYMTEST_TRAIN_AVELENPOLICY_H_ */
