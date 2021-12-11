/*
 * a3cupdater.hpp
 *
 *  Created on: Dec 2, 2021
 *      Author: zf
 */

#ifndef INC_A3C_A3CUPDATER_HPP_
#define INC_A3C_A3CUPDATER_HPP_

#include <torch/torch.h>

#include "a3cgradque.h"

template <typename NetType, typename OptType>
class A3CNetUpdater {
private:
	NetType& net;
	OptType& opt;

	A3CGradQueue& q;

//	TensorBoardLogger tLogger;
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("a3cupdater");

public:
	A3CNetUpdater(NetType& iNet, OptType& iOpt, A3CGradQueue& iQ); //TODO tbLogger
	~A3CNetUpdater() = default;
	A3CNetUpdater(const A3CNetUpdater&) = delete;

	void processGrad();
};

template<typename NetType, typename OptType>
A3CNetUpdater<NetType, OptType>::A3CNetUpdater(NetType& iNet, OptType& iOpt, A3CGradQueue& iQ):
	net(iNet),
	opt(iOpt),
	q(iQ)
{

}

template<typename NetType, typename OptType>
void A3CNetUpdater<NetType, OptType>::processGrad() {
	while (true) {
		std::vector<torch::Tensor> ts = q.pop();

		opt.zero_grad(); //TODO: necessary?
		std::vector<torch::Tensor> params = net.parameters();
		for (int i = 0; i < params.size(); i ++) {
			if (ts[i].numel() == 0) {
				LOG4CXX_DEBUG(logger, "No grad of layer " << i);
				continue;
			}

			params[i].mutable_grad() = ts[i].to(params[i].device());
		}

		torch::nn::utils::clip_grad_norm_(net.parameters(), 0.1);
		opt.step();
	}
}



#endif /* INC_A3C_A3CUPDATER_HPP_ */
