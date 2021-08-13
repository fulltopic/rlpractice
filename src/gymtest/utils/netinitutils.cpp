/*
 * netinitutils.cpp
 *
 *  Created on: May 13, 2021
 *      Author: zf
 */



#include "gymtest/utils/netinitutils.h"


void NetInitUtils::Init_weights(torch::OrderedDict<std::string, torch::Tensor> parameters,
	                  double weight_gain,
	                  double bias_gain,
					  NetInitUtils::InitType initType) {
    for (const auto &parameter : parameters)
    {
        if (parameter.value().size(0) != 0)
        {
            if (parameter.key().find("bias") != std::string::npos)
            {
                torch::nn::init::constant_(parameter.value(), bias_gain);
            }
            else if (parameter.key().find("weight") != std::string::npos)
            {
            	if (initType == NetInitUtils::Kaiming) {
            		torch::nn::init::kaiming_uniform_(parameter.value());
            	} else if (initType == NetInitUtils::Xavier) {
            		torch::nn::init::xavier_normal_(parameter.value());
            	}
            }
        }
    }
}

torch::Tensor NetInitUtils::Orthogonal_(torch::Tensor tensor, double gain) {
    torch::NoGradGuard guard;

//    AT_CHECK(
//        tensor.ndimension() >= 2,
//        "Only tensors with 2 or more dimensions are supported");

    const auto rows = tensor.size(0);
    const auto columns = tensor.numel() / rows;
    auto flattened = torch::randn({rows, columns});

    if (rows < columns)
    {
        flattened.t_();
    }

    // Compute the qr factorization
    torch::Tensor q, r;
    std::tie(q, r) = torch::qr(flattened);
    // Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    auto d = torch::diag(r, 0);
    auto ph = d.sign();
    q *= ph;

    if (rows < columns)
    {
        q.t_();
    }

    tensor.view_as(q).copy_(q);
    tensor.mul_(gain);

    return tensor;
}
