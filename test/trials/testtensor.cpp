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
//#include <random.h>

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

void testSqrt() {
	int inFeatures = 16;
	float stdValue = (float)sqrt(3 / (float)inFeatures);
	std::cout << "stdValue: " << stdValue << std::endl;

	torch::Tensor t = torch::zeros({4, 4});
	torch::nn::init::uniform_(t, -stdValue, stdValue);
	std::cout << "uniform value: " << t << std::endl;
}

void testTo() {
	const int num = 10;
	const int upper = 256;
	const torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);


	torch::Tensor origInt = torch::randint(0, upper, {1, num});
	torch::Tensor intT = origInt.to(torch::kByte);
	torch::Tensor intBT = intT.to(torch::kFloat);
	std::cout << "orig: " << origInt << std::endl;
	std::cout << "int: " << intT << std::endl;
	std::cout << "convert: " << intBT << std::endl;

	torch::Tensor origBool = torch::randint(0, 2, {1, num});
	torch::Tensor BoolT = origBool.to(torch::kByte);
	torch::Tensor boolBT = BoolT.to(torch::kFloat);
	std::cout << "orig: " << origBool << std::endl;
	std::cout << "int: " << BoolT << std::endl;
	std::cout << "convert: " << boolBT << std::endl;

	torch::Tensor origLong = torch::randint(0, 10, {1, num}, longOpt);
	torch::Tensor longT = origLong.to(torch::kByte);
	torch::Tensor longBT = longT.to(torch::kLong);
	std::cout << "orig: " << origLong << std::endl;
	std::cout << "int: " << longT << std::endl;
	std::cout << "convert: " << longBT << std::endl;
}


void testTensorAssign() {
	const int num = 10;
	const torch::TensorOptions byteOpt = torch::TensorOptions().dtype(torch::kByte);

	std::vector<float> dones {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
	std::vector<int64_t> actions {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

	torch::Tensor doneTensor =  torch::zeros({num, 1}, byteOpt);
	torch::Tensor actionTensor = torch::zeros({num, 1}, byteOpt);

	for (int i = 0; i < num; i ++) {
		doneTensor[i][0] = dones[i];
		actionTensor[i][0] = actions[i];
	}
	std::cout << "Assigned " << std::endl;
	std::cout << "done: " << doneTensor << std::endl;
	std::cout << "action: " << actionTensor << std::endl;

	torch::Tensor fDoneTensor = doneTensor.to(torch::kFloat);
	torch::Tensor fActionTensor = actionTensor.to(torch::kFloat);
	std::cout << "To " << std::endl;
	std::cout << "done: " << fDoneTensor << std::endl;
	std::cout << "action: " << fActionTensor << std::endl;
}

void testDist() {
	std::cout << "-------------------> testDist" << std::endl;

	int batchSize = 2;
	int actionNum = 3;
	int atomNum = 4;
	float vMax = 1;
	float vMin = -1;
	float deltaZ = (vMax - vMin) / ((float)(atomNum - 1));

	torch::Tensor valueItem = torch::linspace(vMin, vMax, atomNum);
	std::cout << "valueItems: " << valueItem << std::endl;

	torch::Tensor data = torch::rand({batchSize, actionNum, atomNum});
	torch::Tensor dataProbs = data.softmax(-1);
	std::cout << "dataProbs: " << dataProbs << std::endl;

	torch::Tensor dataDist = dataProbs * valueItem;
	std::cout << "dataDist: " << dataDist << std::endl;

	torch::Tensor dataSum = dataDist.sum(-1, false);
	std::cout << "dataSum: " << dataSum << std::endl;

	auto dataMaxOutput = dataSum.max(-1, true);
	torch::Tensor dataMax = std::get<0>(dataMaxOutput);
	torch::Tensor dataMaxIndice = std::get<1>(dataMaxOutput);
	std::cout << "dataMax: " << dataMax << std::endl;
	std::cout << "dataMaxIndice: " << dataMaxIndice << std::endl;

	dataMaxIndice = dataMaxIndice.unsqueeze(1).expand({batchSize, 1, atomNum});
	std::cout << "Expanded dataMaxIndiec: " << dataMaxIndice << std::endl;
	auto dataPick = dataProbs.gather(1, dataMaxIndice).squeeze(1);
	std::cout << "dataPick: " << dataPick << std::endl;

	torch::Tensor reward = torch::ones({batchSize, 1});
	torch::Tensor mask = torch::ones({batchSize, 1});
	torch::Tensor shift = reward + mask * valueItem;
	std::cout << "shift: " << shift << std::endl;

	shift = shift.clamp(vMin, vMax);
	std::cout << "shift after adjust: " << shift << std::endl;
	auto shiftIndice = (shift - vMin) / deltaZ;
	std::cout << "shift raw indice: " << shiftIndice << std::endl;
	auto l = shiftIndice.floor();
	auto u = shiftIndice.ceil();
	auto lIndice = l.to(torch::kLong);
	auto uIndice = u.to(torch::kLong);
	std::cout << "l: " << l << std::endl;
	std::cout << "u: " << u << std::endl;

	//TODO: delta not right, should be ml += (u - shift), mu += (shift - l)
	auto uDelta = shiftIndice - l;
	auto lDelta = u - shiftIndice;
	std::cout << "lDelta: " << lDelta << std::endl;
	std::cout << "uDelta: " << uDelta << std::endl;
	//if lIndex == rIndex
	auto eqIndice = lIndice.eq(uIndice).to(torch::kFloat);
	std::cout << "eqIndice: " << eqIndice << std::endl;
	lDelta.add_(eqIndice);
	std::cout << "lDelta after adjust: " << lDelta << std::endl;

//	auto lPick = dataPick.gather(-1, lIndice);
//	auto uPick = dataPick.gather(-1, uIndice);
//	std::cout << "lPick: " << lPick << std::endl;
//	std::cout << "uPick: " << uPick << std::endl;
//	std::cout << "lPick sum " << lPick.sum(-1) << std::endl;
//	std::cout << "uPick sum " << uPick.sum(-1) << std::endl;
//
//	auto lDist = lPick * lDelta;
//	auto uDist = uPick * uDelta;
	auto lDist = dataPick * lDelta;
	auto uDist = dataPick * uDelta;
	std::cout << "lDist: " << lDist << std::endl;
	std::cout << "uDist: " << uDist << std::endl;

	auto offset = (torch::linspace(0, batchSize - 1, batchSize) * atomNum).unsqueeze(-1).to(torch::kLong);
	std::cout << "offset: " << offset << std::endl;

	auto nextDist = torch::zeros({batchSize * atomNum});
	lIndice = (lIndice + offset).view({batchSize * atomNum});
	std::cout << "lIndice as vector: " << lIndice << std::endl;
	lDist = lDist.view({batchSize * atomNum});

	nextDist.index_add_(0, lIndice, lDist);
	std::cout << "nextDist after l: " << nextDist << std::endl;

	uIndice = (uIndice + offset).view({batchSize * atomNum});
	std::cout << "uIndice as vector: " << uIndice << std::endl;
	uDist = uDist.view({batchSize * atomNum});
	nextDist.index_add_(0, uIndice, uDist);
	std::cout << "nextDist after r: " << nextDist << std::endl;

	nextDist = nextDist.view({batchSize, atomNum});
	auto sum = nextDist.sum(-1);
	std::cout << "sum: " << sum << std::endl;

	auto maxAction = std::get<1>(dataSum.max(-1));
	std::cout << "max actions: " << maxAction << std::endl;
}

void testMin() {
	int batch = 4;
	int action = 4;
	torch::Tensor q0 = torch::randn({batch, action});
	torch::Tensor q1 = torch::randn({batch, action});
	torch::Tensor q =  torch::min(q0, q1);

	std::cout << "q0: \n " << q0 << std::endl;
	std::cout << "q1: \n " << q1 << std::endl;
	std::cout << "q2: \n " << q << std::endl;
}
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

//	testNorm();
//	testSqrt();

//	testTo();

//	testTensorAssign();
//	testDist();

	testMin();

	return 0;
}
