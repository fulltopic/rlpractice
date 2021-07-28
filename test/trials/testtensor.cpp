/*
 * testtensor.cpp
 *
 *  Created on: Apr 6, 2021
 *      Author: zf
 */


#include <torch/torch.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

namespace {
void testExpandAs() {
//	torch::Tensor a = torch::ones({2, 2});
//	std::cout << "a = " << a << std::endl;
//	torch::Tensor b = torch::zeros({2, 1});
//	b[1][0] = 1;
//	auto c = b.expand_as(a);
//	std::cout << "b = " << b << std::endl;
//	std::cout << "c = " << c << std::endl;
//	auto d = a * (b.expand_as(a));
//	std::cout << "d = " << d << std::endl;

	torch::Tensor t = torch::zeros({2, 1});
	t[0][0] = 1;
	std::cout << "t = " << t << std::endl;
	torch::Tensor a = torch::zeros({2, 2});
	a = t.expand_as(a);
	std::cout << "a = " << a << std::endl;
	torch::Tensor b = torch::zeros({2, 2, 2});
	b = t.expand_as(b);
	std::cout << "b = " << b << std::endl;
	torch::Tensor c = torch::zeros({2, 2, 2, 2});
	c = t.expand_as(c);
	std::cout << "c = " << c << std::endl;
}

void testAssign() {
	torch::Tensor a = torch::ones({2, 2, 2, 2});
	std::cout << "a = " << a << std::endl;
	a[0] = torch::zeros({2, 2, 2});
	std::cout << "aa = " << a << std::endl;
}

void testDataPtr() {
	torch::Tensor a = torch::zeros({2, 2});
	float* data = a.data_ptr<float>();
	data[2] = 1;
	std::cout << "a cpu " << a << std::endl;
	torch::Tensor b = a.to(torch::kCUDA);
	std::cout << "b cuda " << b << std::endl;
}

void testCuda() {
	  if (torch::cuda::is_available()) {
		  std::cout << "CUDA is available" << std::endl;
	  } else {
		  std::cout << "CUDA NOT available" << std::endl;
	  }

	  std::vector<float> dataVec{1, 2, 3, 4, 5, 6};
	  torch::Tensor tensor = torch::from_blob(dataVec.data(), {2, 3});
	  tensor = tensor.to(torch::kCUDA);
	  std::cout << "tensor " << tensor << std::endl;
}

template<typename T>
void printVec(std::vector<T>& datas, std::string cmm) {
	std::cout << cmm << ": ";
	for (const auto& data: datas) {
		std::cout << data << ", ";
	}
	std::cout << std::endl;
}

template<typename T>
void printPtr(const T* datas, const int size, std::string cmm) {
	std::cout << cmm << ": ";
	for (int i = 0; i < size; i ++) {
		std::cout << datas[i] << ", ";
	}
	std::cout << std::endl;
}

void testLong() {
	torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);
	torch::Tensor a = torch::ones({2, 2}, longOpt);
	std::cout << "a: " << a << std::endl;
	std::vector<int64_t> b(a.data_ptr<int64_t>(), a.data_ptr<int64_t>() + a.numel());
	printVec(b, "b");

	std::vector<int64_t> c{2, 3, 1, 2};
	torch::Tensor d = torch::from_blob(c.data(), {2, 2}, longOpt);
	std::cout << "d: " << d << std::endl;
}

void testRandInt() {
	int batchSize = 32;

	auto intOpt = torch::TensorOptions().dtype(torch::kInt);
	auto randActionTensor = torch::randint(0, 18, {batchSize}, intOpt);
	std::cout << "tensor: " << randActionTensor << std::endl;
	int32_t* randActions = randActionTensor.data_ptr<int32_t>();
	printPtr(randActions, batchSize, "int32 actions:");
//	int64_t* ra = randActionTensor.data_ptr<int64_t>(); //error
//	printPtr(ra, batchSize, "int64 actions"); //error
}

void testReserve() {
	std::vector<float> a;
	std::vector<float> b(7, 3);
	std::cout << "a " << a.size() << std::endl;
	std::cout << "b " << b.size() << std::endl;

	a.resize(b.size());
	std::cout << "a after reserve " << a.size() << std::endl;
	std::copy(b.begin(), b.end(), a.begin());
//	a.insert(a.end(), b.begin(), b.end());
	std::cout << "a after insert " << a.size() << std::endl;
	auto c = torch::from_blob(a.data(), {(int)b.size(), 1});
	std::cout << "tensor: " << c << std::endl;

	std::vector<float> aa;
	std::vector<float> bb(7, 4);
	std::cout << "a " << aa.size() << std::endl;
	std::cout << "b " << bb.size() << std::endl;

	aa.reserve(bb.size());
	std::cout << "a after reserve " << aa.size() << std::endl;
	std::copy(bb.begin(), bb.end(), aa.begin());
//	a.insert(a.end(), b.begin(), b.end());
	std::cout << "a after insert " << aa.size() << std::endl;
	auto cc = torch::from_blob(aa.data(), {(int)bb.size(), 1});
	std::cout << "tensor: " << cc << std::endl;
	std::cout << "a after aa: " << a << std::endl;
	std::cout << "c after aa: " << c << std::endl;
//	std::vector<float>
}

void testDiff() {
	auto t0 = torch::randn({2, 2, 2});
	auto t1 = torch::randn({2, 2, 2});
	auto t2 = t0 - t1;
	std::cout << "t2 = " << t2 << std::endl;
	auto t3 = t2.count_nonzero();
	std::cout << "diff " << t3 << std::endl;
	auto t4 = t3.sum();
	std::cout << "t4 = " << t4 << std::endl;
}

void testExpand() {
	auto a = torch::randn({2, 2});
	auto b = torch::randn({2, 1});
	auto c = torch::randn({2, 1});
	std::cout << "a = " << a << std::endl;
	std::cout << "b = " << b << std::endl;
	std::cout << "c = " << c << std::endl;

	auto d = b + c;
	std::cout << "d = " << d << std::endl;
//	auto e = d.expand(0, 2);
//	std::cout << "e = " << e << std::endl;
	auto f = a + d;
	std::cout << "f = " << f << std::endl;
}

void testClamp() {
	auto a = torch::randn({2, 2});
	std::cout << "a = " << a << std::endl;
	auto b = a.clamp(-1, 1);
	std::cout << "b = " << b << std::endl;
	std::cout << "a after clamp: " << a << std::endl;
}

void testCategory() {
	int catNum = 3;
	auto action = torch::randn({1, catNum});
	auto aProb = torch::softmax(action, -1);
	std::random_device rd;
	std::mt19937 gen(rd());
	std::vector<float> da(aProb.data_ptr<float>(), aProb.data_ptr<float>() + catNum);
	std::vector<float> pos {0.1, 0.1, 0.8};
	std::discrete_distribution<int> d(da.begin(), da.end());
	std::cout << "aProb: " << aProb << std::endl;
	std::vector<int> m(catNum, 0);
	for(int n=0; n<10000; ++n) {
		++m[d(gen)];
	}
	for(auto p : m) {
		std::cout << " generated " << p << " times\n";
	}
}

void testMean() {
	auto a = torch::ones({2, 2});
	auto b = torch::ones({2, 2}) * 3;
	auto c = a + b;
	auto cSum = c.sum();
	std::cout << cSum << std::endl;
	auto cMean = c.mean();
	std::cout << cMean << std::endl;

	auto d = a.view({2, 2, 1});
	auto e = b.view({2, 2, 1});
	auto f = d + e;
	auto fSum = f.sum();
	std::cout << fSum << std::endl;
	auto fMean = f.mean();
	std::cout << fMean << std::endl;
}

void testEntropy() {
	int actNum = 6;
	auto a = torch::ones({4, actNum}).div(actNum);
	auto b = torch::softmax(a, -1);
	auto c = torch::log_softmax(a, -1);
	auto d = b * c;
	auto dSum = d.sum();
	auto dMean = d.mean();

	std::cout << "a = " << a << std::endl;
	std::cout << "b = " << b << std::endl;
	std::cout << "c = " << c << std::endl;
	std::cout << "d = " << d << std::endl;
	std::cout << "e = " << dSum << std::endl;
	std::cout << "f = " << dMean << std::endl;
}

void testMultinomial() {
	int clientNum = 4;
	int probLen = 9;
	std::vector<std::vector<int>> nums(clientNum, std::vector<int>(probLen, 0));
	torch::Tensor probs = torch::randn({clientNum, probLen}).softmax(-1);
	std::cout << "probs: " << probs << std::endl;
	for (int i = 0; i < 100; i ++) {
		torch::Tensor samples = probs.multinomial(1);
//		std::cout << "samples: " << samples << std::endl;
		std::vector<long> sampleVec(samples.data_ptr<long>(), samples.data_ptr<long>() + clientNum);
		for (int j = 0; j < sampleVec.size(); j ++) {
			nums[j][sampleVec[j]] ++;
		}
	}
	std::cout << "samples: " << std::endl;
	for (int i = 0; i < clientNum; i ++) {
		std::cout << nums[i] << std::endl;
	}
}

void testBinomial() {
	float epsilon = 0.9;
	const torch::Tensor probTensor(torch::ones({1}) *  epsilon);
	const torch::Tensor countTensor = torch::ones({1});

	int count0 = 0;
	int count1 = 1;

	for (int i = 0; i < 100; i ++) {
		torch::Tensor greedyTensor = torch::binomial(countTensor, probTensor);
		int greedyValue = greedyTensor.item().toInt();

		if (greedyValue == 1) {
			count1 ++;
		} else {
			count0 ++;
		}
	}

	std::cout << "count0 = " << count0 << " count1 = " << count1 << std::endl;
}

void testLoss() {
//	torch::nn::HuberLoss huberLossComputer = torch::nn::HuberLoss();
//	torch::Tensor output = torch::randn({4, 1});
//	torch::Tensor target = torch::randn({4, 1});
//
//	torch::Tensor loss0 = huberLossComputer(output, target);
//	torch::Tensor loss1 = huberLossComputer->forward(output, target);
//	torch::Tensor loss2 = torch::nn::functional::huber_loss(output, target);
//
//	std::cout << "loss0 = " << loss0 << std::endl;
//	std::cout << "loss1 = " << loss1 << std::endl;
//	std::cout << "loss2 = " << loss2 << std::endl;
//
//	output = torch::randn({4, 1});
//	target = torch::randn({4, 1});
//
//	loss0 = huberLossComputer(output, target);
//	loss1 = huberLossComputer->forward(output, target);
//	loss2 = torch::nn::functional::huber_loss(output, target);
//
//	std::cout << "loss0 = " << loss0 << std::endl;
//	std::cout << "loss1 = " << loss1 << std::endl;
//	std::cout << "loss2 = " << loss2 << std::endl;

	torch::nn::HuberLoss huberLossComputer = torch::nn::HuberLoss();

	torch::Tensor ws0 = torch::randn({4, 1});
	torch::Tensor ws1 = torch::zeros({4, 1});
	torch::Tensor ws2 = torch::zeros({4, 1});
	ws1.copy_(ws0);
	ws2.copy_(ws0);
	ws0.requires_grad_();
	ws1.requires_grad_();
	ws2.requires_grad_();

	torch::Tensor target = torch::randn({4, 1}).requires_grad_();

	torch::Tensor input0 = torch::ones({4, 1});
	torch::Tensor input1 = torch::ones({4, 1});
	torch::Tensor input2 = torch::ones({4, 1});

	torch::Tensor output0 = ws0 * input0;
	output0.requires_grad_();
	torch::Tensor output1 = ws1 * input1;
	output1.requires_grad_();
	torch::Tensor output2 = ws2 * input2;
	output2.requires_grad_();

	torch::Tensor loss0 = huberLossComputer(output0, target);
	torch::Tensor loss1 = huberLossComputer->forward(output1, target);
	torch::Tensor loss2 = torch::nn::functional::huber_loss(output2, target);

	loss0.backward();
	std::cout << "ws grad " << ws0.grad() << std::endl;
	loss1.backward();
	std::cout << "ws grad " << ws1.grad() << std::endl;
	loss2.backward();
	std::cout << "ws grad " << ws2.grad() << std::endl;

	input0 = torch::ones({4, 1}) * 0.9;
	input1 = torch::ones({4, 1}) * 0.9;
	input2 = torch::ones({4, 1}) * 0.9;

	torch::Tensor output3 = ws0 * input0;
	output3.requires_grad_();
	torch::Tensor output4 = ws1 * input1;
	output4.requires_grad_();
	torch::Tensor output5 = ws2 * input2;
	output5.requires_grad_();

	torch::Tensor loss3 = huberLossComputer(output3, target);
	torch::Tensor loss4 = huberLossComputer->forward(output4, target);
	torch::Tensor loss5 = torch::nn::functional::huber_loss(output5, target);

	loss3.backward();
	std::cout << "ws grad " << ws0.grad() << std::endl;
	loss4.backward();
	std::cout << "ws grad " << ws1.grad() << std::endl;
	loss5.backward();
	std::cout << "ws grad " << ws2.grad() << std::endl;
}

void testmse() {
	torch::nn::HuberLoss huberLossComputer = torch::nn::HuberLoss();
	torch::nn::MSELoss mseLossComputer = torch::nn::MSELoss();


	torch::Tensor ws0 = torch::randn({4, 1});
	torch::Tensor ws1 = torch::zeros({4, 1});
	torch::Tensor ws2 = torch::zeros({4, 1});
	ws1.copy_(ws0);
	ws2.copy_(ws0);
	ws0.requires_grad_();
	ws1.requires_grad_();
	ws2.requires_grad_();

	torch::Tensor target = torch::randn({4, 1}).requires_grad_();

	torch::Tensor input0 = torch::ones({4, 1});
	torch::Tensor input1 = torch::ones({4, 1});
	torch::Tensor input2 = torch::ones({4, 1});

	torch::Tensor output0 = ws0 * input0;
	output0.requires_grad_();
	torch::Tensor output1 = ws1 * input1;
	output1.requires_grad_();
	torch::Tensor output2 = ws2 * input2;
	output2.requires_grad_();

//	torch::Tensor loss0 = huberLossComputer(output0, target);
	torch::Tensor loss0 = (output0 - target).pow(2).mean();
	torch::Tensor loss1 = mseLossComputer->forward(output1, target);
	torch::Tensor loss2 = torch::nn::functional::mse_loss(output2, target);

	loss0.backward();
	std::cout << "ws grad " << ws0.grad() << std::endl;
	loss1.backward();
	std::cout << "ws grad " << ws1.grad() << std::endl;
	loss2.backward();
	std::cout << "ws grad " << ws2.grad() << std::endl;
}

void testfromblob() {
	std::vector<float> raws{5.64432e-40, 0};
	torch::Tensor t = torch::from_blob(raws.data(), {2, 1});

	std::cout << "raws = " << raws << std::endl;
	std::cout << "t = " << t << std::endl;
}

void testChunk() {
	const int maxStep = 3;
	const int roundNum = 4;
	const int trajStep = maxStep * roundNum;
	const int envNum = 8;

	std::vector<float> data(trajStep * envNum, 0);
	for (int i = 0; i < trajStep; i ++) {
		for (int j = 0; j < envNum; j ++) {
			data[i * envNum + j] = i * 10 + j;
		}
	}
	std::cout << "data: " << std::endl << data << std::endl;

	torch::Tensor bulkTensor = torch::from_blob(data.data(), {trajStep, envNum, 1});
	auto rs = torch::split(bulkTensor.view({roundNum * maxStep * envNum, 1}), maxStep * envNum);
	std::cout << rs.size() << " parts " << std::endl;
	auto r = rs[0];
	std::cout << "a chunk " << r << std::endl;
//	std::cout << "a chunk " << rs[1] << std::endl;
	std::cout << "a chunk " << rs[2] << std::endl;
}

void testNorm() {
	std::vector<float> data {
		-4.3510,
		 -4.7648,
		 -4.5042,
		 -4.6648,
		 -4.2306,
		 -4.0069,
		 -3.8417,
		 -4.0403,
		 -3.5070,
		 -3.6849,
		 -2.9249,
		 -3.1565,
		 -1.8906,
		 -1.5126,
		 -2.0397,
		 -1.8644
	};

	torch::Tensor t = torch::from_blob(data.data(), {16, 1});
	auto m = t.mean();
	std::cout << "m = " << m << std::endl;
	auto s = t.std();
	std::cout << "s = " << s << std::endl;

	auto u = t - m;
	std::cout << "u = " << u << std::endl;

	auto f = u / s;
	std::cout << "f = " << f << std::endl;
}
//void testDevice() {
//	auto device = torch::device("cuda:0");
//	std::cout << "device " << device << std::endl;
//}
}

int main() {
//	testExpandAs();
//	testAssign();
//	testDataPtr();
//	testCuda();
//	testLong();
//	testRandInt();
//	testReserve();
//	testDiff();
//	testExpand();
//	testClamp();
//	testMean();
//	testEntropy();
//	testCategory();
//	testMultinomial();
//	testBinomial();
//	testLoss();
//	testmse();
//	testfromblob();
//	testDevice();
//	testChunk();

	testNorm();

	return 0;
}
