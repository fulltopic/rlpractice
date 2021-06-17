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
}

int main() {
//	testXGrad();
	testBatchGrad();
}
