/*
 * cartqnet.cpp
 *
 *  Created on: Aug 23, 2021
 *      Author: zf
 */




/*
 * cartnet.cpp
 *
 *  Created on: Apr 24, 2021
 *      Author: zf
 */

#include "gymtest/lunarnets/netconfig.h"
#include <torch/torch.h>
#include "gymtest/lunarnets/cartqnet.h"

#include <string.h>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>

//void init_weights(const torch::OrderedDict<std::string, torch::Tensor>& parameters,
//                  double weight_gain,
//                  double bias_gain)
//{
//    for (const auto &parameter : parameters)
//    {
//        if (parameter.value().size(0) != 0)
//        {
//            if (parameter.key().find("bias") != std::string::npos)
//            {
//                torch::nn::init::constant_(parameter.value(), bias_gain);
//            }
//            else if (parameter.key().find("weight") != std::string::npos)
//            {
//                torch::nn::init::orthogonal_(parameter.value(), weight_gain);
//            }
//        }
//    }
//}

CartFcQNet::CartFcQNet(int input, int output):
	inputNum(input),
	outputNum(output),
	fc0(input, 256),
	fcOutput(256, output) {
	register_module("fc0", fc0);
	register_module("fcOutput", fcOutput);

//	init_weights(fc0->named_parameters(), sqrt(2.0), 0);
//	init_weights(fcOutput->named_parameters(), sqrt(2.0), 0);
}

torch::Tensor CartFcQNet::forward(torch::Tensor input) {
	torch::Tensor output = fc0->forward(input);
	output = torch::relu(output);
	output = fcOutput->forward(output);
//	output = torch::relu(output);

	return output;
}



