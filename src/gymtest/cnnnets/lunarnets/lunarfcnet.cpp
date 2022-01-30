/*
 * fcnet.cpp
 *
 *  Created on: Apr 4, 2021
 *      Author: zf
 */



#include "gymtest/cnnnets/lunarnets/netconfig.h"
#include <torch/torch.h>
#include "gymtest/cnnnets/lunarnets/lunarfcnet.h"

LunarFcNet::LunarFcNet():
	fc0(NetConfig::LunarInputW, 64),
	fcOutput(64, NetConfig::LunarOutputW) {
	register_module("fc0", fc0);
	register_module("fcOutput", fcOutput);
}

torch::Tensor LunarFcNet::forward(torch::Tensor input) {
	torch::Tensor output = fc0->forward(input);
	output = torch::relu(output);
	output = fcOutput->forward(output);
	output = torch::relu(output);

	return output;
}
