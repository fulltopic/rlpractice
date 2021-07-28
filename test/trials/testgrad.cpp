/*
 * testgrad.cpp
 *
 *  Created on: Apr 15, 2021
 *      Author: zf
 */



#include <torch/torch.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

namespace {
void testXGrad() {
	torch::Tensor x = torch::ones({2, 2}).requires_grad_();
	torch::Tensor w = (torch::ones({2, 2}).requires_grad_() * 2);

	torch::Tensor y = w * x;
	y.requires_grad_();
	y.sum().backward();

	std::cout << x.grad() << std::endl;
	std::cout << w.grad() << std::endl;
	std::cout << y.grad() << std::endl;
}

void testBatchGrad() {
	torch::Tensor w = torch::ones({4, 1}).requires_grad_();
	torch::Tensor x = torch::randn({2, 4}).requires_grad_();
	torch::Tensor y = torch::matmul(x, w).requires_grad_();

	auto z = y.sum();
	z.backward();

	std::cout << "x = " << x << std::endl;
	std::cout << x.grad() << std::endl;
	std::cout << w.grad() << std::endl;
	std::cout << "y has grad? " << y.grad() << std::endl;
	std::cout << "z has grad? " << z.grad() << std::endl;
}

struct FcNet: torch::nn::Module {
private:
	torch::nn::Linear fc;

public:
	FcNet(): fc(torch::nn::LinearOptions(1, 1).bias(false)) {
//		fc->weight = torch::zeros(fc->weight.sizes()).requires_grad_();
		register_module("fc", fc);
	}
	~FcNet() = default;

	torch::Tensor forward(torch::Tensor input) {
		return fc->forward(input);
	}
//	torch::Tensor getLoss(torch::Tensor input, torch::Tensor returns);
};

void testClip() {
	FcNet fc;
	torch::optim::SGD optimizer(fc.parameters(), torch::optim::SGDOptions(1e-1));
	std::cout << "parameters: " << fc.parameters() << std::endl;

	torch::Tensor x = torch::ones({1, 1});
	torch::Tensor target = torch::ones({1, 1}) * 4;
	torch::Tensor y = fc.forward(x);
	torch::Tensor loss = torch::nn::functional::mse_loss(y, target);
	optimizer.zero_grad();
	loss.backward();
	optimizer.step();
	std::cout << "parameters: " << fc.parameters() << std::endl;

	x = torch::ones({1, 1});
	target = torch::ones({1, 1}) * 4;
	y = fc.forward(x);
	target = target.clip(0, 1);
	loss = torch::nn::functional::mse_loss(y, target);
	optimizer.zero_grad();
	loss.backward();
	optimizer.step();
	std::cout << "parameters: " << fc.parameters() << std::endl;

	x = torch::ones({1, 1});
	target = torch::ones({1, 1}) * 4;
	y = fc.forward(x);
//	target = target.clip(0, 1);
	loss = torch::nn::functional::mse_loss(y, target);
	optimizer.zero_grad();
	loss.backward();
	optimizer.step();
	std::cout << "parameters: " << fc.parameters() << std::endl;
}
}

int main() {
//	testXGrad();
//	testBatchGrad();
	testClip();
}
