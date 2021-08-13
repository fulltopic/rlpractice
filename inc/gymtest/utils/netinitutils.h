/*
 * netinitutils.h
 *
 *  Created on: May 13, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_UTILS_NETINITUTILS_H_
#define INC_GYMTEST_UTILS_NETINITUTILS_H_
#include <torch/torch.h>
#include <string>

class NetInitUtils {
public:
	enum InitType {
		Xavier,
		Kaiming,
	};

	static void Init_weights(torch::OrderedDict<std::string, torch::Tensor> parameters,
	                  double weight_gain,
	                  double bias_gain,
					  InitType initType = Kaiming);

	static torch::Tensor Orthogonal_(torch::Tensor tensor, double gain);

};



#endif /* INC_GYMTEST_UTILS_NETINITUTILS_H_ */
