/*
 * testrtnorm.cpp
 *
 *  Created on: May 14, 2021
 *      Author: zf
 */


#include <torch/torch.h>
#include "gymtest/utils/inputnorm.h"
#include <iostream>
#include <string>

namespace {
void test0() {
	auto inputNorm = InputNorm(torch::kCPU);

	torch::Tensor t0 = torch::randn({3, 3});
	inputNorm.update(t0, 1);
	auto mean = t0.mean().item<float>();
	auto var = t0.var(false).sqrt().item<float>();
//	std::cout << "t0 " << std::endl << t0 << std::endl;
	std::cout << "mean: " << mean << std::endl;
	std::cout << "var: " << var << std::endl;
	std::cout << "norm mean: " << inputNorm.getMean() << std::endl;
	std::cout << "norm var: " << inputNorm.getVar() << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;

	torch::Tensor t1 = torch::randn({3, 3});
	inputNorm.update(t1, 1);
//	std::cout << "t1 " << std::endl << t1 << std::endl;
	std::cout << "norm mean: " << inputNorm.getMean() << std::endl;
	std::cout << "norm var: " << inputNorm.getVar() << std::endl;

	auto sum0 = t0.sum();
	auto sum1 = t1.sum();
	auto sumMean = (sum0 + sum1) / (t0.numel() * 2);
	std::cout << "sumMean: " << sumMean << std::endl;

	auto powT0 = (t0 - sumMean.item<float>()).pow(2).sum();
	auto powT1 = (t1 - sumMean.item<float>()).pow(2).sum();
	auto pow = (powT0 + powT1) / (t0.numel() * 2);
	auto powVar = pow.sqrt();
	std::cout << "powVar: " << powVar << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;

	torch::Tensor t2 = torch::randn({3, 3});
	inputNorm.update(t2, 1);
	std::cout << "norm mean: " << inputNorm.getMean() << std::endl;
	std::cout << "norm var: " << inputNorm.getVar() << std::endl;

	sum0 = t0.sum();
	sum1 = t1.sum();
	auto sum2 = t2.sum();
	sumMean = (sum0 + sum1 + sum2) / (t0.numel() * 3);
	std::cout << "sumMean: " << sumMean << std::endl;

	powT0 = (t0 - sumMean.item<float>()).pow(2).sum();
	powT1 = (t1 - sumMean.item<float>()).pow(2).sum();
	auto powT2 = (t2 - sumMean.item<float>()).pow(2).sum();
	pow = (powT0 + powT1 + powT2) / (t0.numel() * 3);
	powVar = pow.sqrt();
	std::cout << "powVar: " << powVar << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;

	torch::Tensor t3 = torch::randn({3, 3});
	inputNorm.update(t3, 1);
	std::cout << "norm mean: " << inputNorm.getMean() << std::endl;
	std::cout << "norm var: " << inputNorm.getVar() << std::endl;

	sum0 = t0.sum();
	sum1 = t1.sum();
	sum2 = t2.sum();
	auto sum3 = t3.sum();
	sumMean = (sum0 + sum1 + sum2 + sum3) / (t0.numel() * 4);
	std::cout << "sumMean: " << sumMean << std::endl;

	powT0 = (t0 - sumMean.item<float>()).pow(2).sum();
	powT1 = (t1 - sumMean.item<float>()).pow(2).sum();
	powT2 = (t2 - sumMean.item<float>()).pow(2).sum();
	auto powT3 = (t3 - sumMean.item<float>()).pow(2).sum();
	pow = (powT0 + powT1 + powT2 + powT3) / (t0.numel() * 4);
	powVar = pow.sqrt();
	std::cout << "powVar: " << powVar << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;
}


void test1() {
	auto inputNorm = InputNorm(torch::kCPU);

	torch::Tensor t0 = torch::randn({3, 3});
	inputNorm.update(t0, 1);
	auto mean = t0.mean().item<float>();
	auto var = t0.var().sqrt().item<float>();
//	std::cout << "t0 " << std::endl << t0 << std::endl;
	std::cout << "mean: " << mean << std::endl;
	std::cout << "var: " << var << std::endl;
	std::cout << "norm mean: " << inputNorm.getMean() << std::endl;
	std::cout << "norm var: " << inputNorm.getVar() << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;

	torch::Tensor t1 = torch::randn({3, 3});
	inputNorm.update(t1, 1);
	auto datas = torch::cat({t0, t1}, 0);
	mean = datas.mean().item<float>();
	var = datas.var().sqrt().item<float>();
//	std::cout << "t0 " << std::endl << t0 << std::endl;
	std::cout << "mean: " << mean << std::endl;
	std::cout << "var: " << var << std::endl;
	std::cout << "norm mean: " << inputNorm.getMean() << std::endl;
	std::cout << "norm var: " << inputNorm.getVar() << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;
}

void test2() {
	torch::Tensor a = torch::ones({2, 2});
	auto b = a.pow(2).sum();
	a = a * 2;
	auto d = a.pow(2).sum();
	std::cout << "b = " << b << std::endl;
	std::cout << "d = " << d << std::endl;
}

void test3() {
	auto inputNorm = InputNorm(torch::kCPU);

	torch::Tensor t0 = torch::eye(2);
	std::cout << "t0 = " << t0 << std::endl;
	inputNorm.update(t0, 1);
	auto mean = t0.mean().item<float>();
	auto var = t0.var(false).sqrt().item<float>();
//	std::cout << "t0 " << std::endl << t0 << std::endl;
	std::cout << "mean: " << mean << std::endl;
	std::cout << "var: " << var << std::endl;
	std::cout << "norm mean: " << inputNorm.getMean() << std::endl;
	std::cout << "norm var: " << inputNorm.getVar() << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;

	torch::Tensor t1 = torch::eye(2);
	inputNorm.update(t1, 1);
//	std::cout << "t1 " << std::endl << t1 << std::endl;
	std::cout << "norm mean: " << inputNorm.getMean() << std::endl;
	std::cout << "norm var: " << inputNorm.getVar() << std::endl;

	auto sum0 = t0.sum();
	auto sum1 = t1.sum();
	auto sumMean = (sum0 + sum1) / (t0.numel() * 2);
	std::cout << "sumMean: " << sumMean << std::endl;

	auto powT0 = (t0 - sumMean.item<float>()).pow(2).sum();
	auto powT1 = (t1 - sumMean.item<float>()).pow(2).sum();
	auto pow = (powT0 + powT1) / (t0.numel() * 2);
	auto powVar = pow.sqrt();
	std::cout << "powVar: " << powVar << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;

	torch::Tensor t2 = torch::eye(2);
	inputNorm.update(t2, 1);
	std::cout << "norm mean: " << inputNorm.getMean() << std::endl;
	std::cout << "norm var: " << inputNorm.getVar() << std::endl;

	sum0 = t0.sum();
	sum1 = t1.sum();
	auto sum2 = t2.sum();
	sumMean = (sum0 + sum1 + sum2) / (t0.numel() * 3);
	std::cout << "sumMean: " << sumMean << std::endl;

	powT0 = (t0 - sumMean.item<float>()).pow(2).sum();
	powT1 = (t1 - sumMean.item<float>()).pow(2).sum();
	auto powT2 = (t2 - sumMean.item<float>()).pow(2).sum();
	pow = (powT0 + powT1 + powT2) / (t0.numel() * 3);
	powVar = pow.sqrt();
	std::cout << "powVar: " << powVar << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;

	torch::Tensor t3 = torch::eye(2);
	inputNorm.update(t3, 1);
	std::cout << "norm mean: " << inputNorm.getMean() << std::endl;
	std::cout << "norm var: " << inputNorm.getVar() << std::endl;

	sum0 = t0.sum();
	sum1 = t1.sum();
	sum2 = t2.sum();
	auto sum3 = t3.sum();
	sumMean = (sum0 + sum1 + sum2 + sum3) / (t0.numel() * 4);
	std::cout << "sumMean: " << sumMean << std::endl;

	powT0 = (t0 - sumMean.item<float>()).pow(2).sum();
	powT1 = (t1 - sumMean.item<float>()).pow(2).sum();
	powT2 = (t2 - sumMean.item<float>()).pow(2).sum();
	auto powT3 = (t3 - sumMean.item<float>()).pow(2).sum();
	pow = (powT0 + powT1 + powT2 + powT3) / (t0.numel() * 4);
	powVar = pow.sqrt();
	std::cout << "powVar: " << powVar << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;
}

void test4() {
	auto inputNorm = InputNorm(torch::kCPU);

	torch::Tensor t0 = torch::eye(4);
	inputNorm.update(t0, 1);
	inputNorm.save("testnorm_test4");

	std::cout << "mean = " << inputNorm.getMean() << ", var = " << inputNorm.getVar() << std::endl;
}

void test5() {
	auto inputNorm = InputNorm(torch::kCPU);

//	torch::Tensor t0 = torch::eye(4);
//	inputNorm.udpate(t0, 1);
	inputNorm.load("testnorm_test4");

	std::cout << "mean = " << inputNorm.getMean() << ", var = " << inputNorm.getVar() << std::endl;
}

void testV() {
	const long trajLen = 4;

	const float lambda = 0.95;
	const float gamma = 1;

	torch::Tensor rewards = torch::randn({trajLen, 1});
	torch::Tensor values = torch::randn({trajLen, 1});
	torch::Tensor lastValue = torch::randn({1});
	torch::Tensor nextValue = torch::zeros({1});
	torch::Tensor gaeAdvs = torch::zeros({trajLen, 1});
	torch::Tensor vTargets = torch::zeros({trajLen, 1});
	torch::Tensor gaeAdv = torch::zeros({1});
	torch::Tensor vTarget = torch::zeros({1});

	vTarget.copy_(lastValue);
	nextValue.copy_(lastValue);

	for (int i = trajLen - 1; i >= 0; i --) {
		auto delta = rewards[i] + gamma * nextValue - values[i];
		gaeAdv = delta + gamma * lambda * gaeAdv;

		vTarget = rewards[i] + gamma * vTarget;

		gaeAdvs[i].copy_(gaeAdv);
		vTargets[i].copy_(vTarget);

		nextValue = values[i];
	}

	auto vTargetsGae = gaeAdvs + values;

	std::cout << "Plain V target: \n" << vTargets << std::endl;
	std::cout << "Gae V target: \n" << vTargetsGae << std::endl;
}
}


int main() {
//	test0();
//	test1();
//	test3();

//	test4();
//	test5();

	testV();
}
