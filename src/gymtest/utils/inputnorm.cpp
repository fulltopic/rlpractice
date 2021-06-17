/*
 * inputnorm.cpp
 *
 *  Created on: May 14, 2021
 *      Author: zf
 */

#include "gymtest/utils/inputnorm.h"

InputNorm::InputNorm(torch::Device dType): deviceType(dType) {
	mean = torch::tensor(0).to(deviceType);
	var = torch::tensor(1e-4).to(deviceType);
}

//TODO: Different mean for different channel
void InputNorm::update(torch::Tensor input, float batchNum) {
	torch::NoGradGuard guard;

	auto inputMean = input.mean();
	auto inputVar = input.var(false);

    auto delta = inputMean - mean;
    auto totalCount = count + batchNum;
    auto newMean = (mean + delta * batchNum / totalCount);

    auto eSquare = count * var + batchNum * inputVar + count * mean.pow(2) + batchNum * inputMean.pow(2);
    auto squareE = newMean.pow(2);
    var = eSquare / (totalCount) - squareE;

    mean = newMean;
    count = totalCount;
//    auto m_a = var * count;
//    auto m_b = inputVar * batchNum;
//    auto m2 = m_a + m_b + torch::pow(delta, 2) * count * batchNum / total_count;
//    var = (m2 / total_count);
//    count = total_count;
}

void InputNorm::save(std::string prefix) {
	std::string path = prefix + "_norm.pt";

	torch::Tensor norms = torch::zeros({3, 1});
	float* normPtr = norms.data_ptr<float>();
	normPtr[0] = mean.item<float>();
	normPtr[1] = var.item<float>();
	normPtr[2] = count;

	torch::save(norms, path);
}

void InputNorm::load(std::string prefix) {
	std::string path = prefix + "_norm.pt";

	torch::Tensor norms = torch::zeros({3, 1});
	torch::load(norms, path);
	float* normPtr = norms.data_ptr<float>();

	mean = torch::tensor(normPtr[0]);
	var = torch::tensor(normPtr[1]);
	count = normPtr[2];
}
