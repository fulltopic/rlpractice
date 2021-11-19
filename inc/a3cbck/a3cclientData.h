/*
 * a3cclientData.h
 *
 *  Created on: Nov 7, 2021
 *      Author: zf
 */

#ifndef INC_A3C_A3CCLIENTDATA_H_
#define INC_A3C_A3CCLIENTDATA_H_

#include "commdata.h"

class A3CClientData: public CommData {
public:
	A3CClientData(std::vector<at::IntArrayRef> shapes);
	virtual ~A3CClientData() = default;
	A3CClientData(const A3CClientData&) = delete;

	virtual void updateGrad(std::vector<torch::Tensor> deltaGrads);
	virtual void syncTarget(std::vector<torch::Tensor> targetParams);
	virtual void resetGrad();
	virtual void resetTargetParam();
protected:
	virtual const std::vector<torch::Tensor>& getGrad();

private:
	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	torch::Tensor cmdTensor = torch::zeros({1}, longOpt);
	torch::Tensor startIndexTensor = torch::zeros({1}, longOpt);
	torch::Tensor endIndexTensor = torch::zeros({1}, longOpt);
};



#endif /* INC_A3C_A3CCLIENTDATA_H_ */
