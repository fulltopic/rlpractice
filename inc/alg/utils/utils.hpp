/*
 * utils.hpp
 *
 *  Created on: Dec 10, 2021
 *      Author: zf
 */

#ifndef INC_ALG_UTILS_HPP_
#define INC_ALG_UTILS_HPP_

#include <torch/torch.h>

#include <vector>

class AlgUtils {
public:
	template<typename NetType>
	static void SyncNet(NetType& srcNet, NetType& dstNet, const float tau) ;
};

//TODO: To check type of NetType
template<typename NetType>
void AlgUtils::SyncNet(NetType& srcNet, NetType& dstNet, const float tau) {
	torch::NoGradGuard guard;

	auto paramDict = srcNet.named_parameters();
	auto buffDict = srcNet.named_buffers();
	auto targetParamDict = dstNet.named_parameters();
	auto targetBuffDict = dstNet.named_buffers();

	for (const auto& item: paramDict) {
		const auto& key = item.key();
		const auto param = item.value();
		auto& targetParam = targetParamDict[key];

		targetParam.mul_(1 - tau);
		targetParam.add_(param, tau);
	}

	for (const auto& item: buffDict) {
		const auto& key = item.key();
		const auto& buff = item.value();
		auto& targetBuff = targetBuffDict[key];

		targetBuff.mul_(1 - tau);
		targetBuff.add_(buff, tau);
	}
}



#endif /* INC_ALG_UTILS_HPP_ */
