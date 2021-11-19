/*
 * commdata.h
 *
 *  Created on: Nov 7, 2021
 *      Author: zf
 */

#ifndef INC_A3C_COMMDATA_H_
#define INC_A3C_COMMDATA_H_

#include <torch/torch.h>
#include <vector>

#include "a3cconfig.h"

class A3CClientHandler;

class CommData {
public:
	virtual ~CommData() = 0;

	virtual void updateGrad(std::vector<torch::Tensor> deltaGrads) = 0;
	virtual void syncTarget(std::vector<torch::Tensor> targetParams) = 0;
	virtual void resetGrad();
	virtual void resetTargetParam();

protected:
	std::vector<torch::Tensor> grads;
	std::vector<torch::Tensor> targetParams;

	virtual const std::vector<torch::Tensor>& getGrad() = 0;

	friend A3CClientHandler;
};



#endif /* INC_A3C_COMMDATA_H_ */
