/*
 * testpackpad.cpp
 *
 *  Created on: Jan 29, 2022
 *      Author: zf
 */


#include <iostream>
#include <vector>
#include <algorithm>

#include <torch/torch.h>

#include "gymtest/cnnnets/lunarnets/cartnet.h"

namespace {
torch::TensorOptions longOpt = torch::TensorOptions().dtype(torch::kLong);

void testPack() {
	const int batch = 4;
	std::vector<torch::Tensor> tensorVec;
	std::vector<long> seqLens;
	for (int i = 0; i < batch; i ++) {
		tensorVec.push_back(torch::ones({(batch - i), 2}) * (batch - i));
		seqLens.push_back(batch - i);

		std::cout << "Orig tensor " << i << std::endl << tensorVec[tensorVec.size() - 1] << std::endl;
	}
	std::cout << "seqLens = " << seqLens << std::endl;

	auto padSeqTensor = torch::nn::utils::rnn::pad_sequence(tensorVec, true);
	std::cout << "padSeqTensor: " << std::endl << padSeqTensor << std::endl;

//	CartFcNet net(2, 4);
//	torch::Tensor output = net.forward(origTensor);
//	std::cout << "output = " << std::endl << output << std::endl;
//	torch::Tensor padTensor = torch::constant_pad_nd(origTensor, {0,})//TODO: pad at each single tensor

	torch::Tensor seqTensor = torch::from_blob(seqLens.data(), {batch}, longOpt);
	std::cout << "seqTensor = " << seqTensor << std::endl;
	auto packSeq = torch::nn::utils::rnn::pack_padded_sequence(padSeqTensor, seqTensor,  true);
	std::cout << "packSeq = " << std::endl << packSeq.batch_sizes() << std::endl
											<< packSeq.sorted_indices() << std::endl
											<< packSeq.unsorted_indices() << std::endl;
	std::cout << "pack data --------------------------------------> " << std::endl
			<< packSeq.data() << std::endl;

	auto padOutput = torch::nn::utils::rnn::pad_packed_sequence(packSeq, true);
	std::cout << "padded " << std::endl << std::get<0>(padOutput) << std::endl;
	std::cout << "paddedSeq " << std::endl << std::get<1>(padOutput) << std::endl;

	torch::Tensor unpackData = std::get<0>(padOutput);
	std::vector<torch::Tensor> unpackVec;
	for (int i = 0; i < seqLens.size(); i ++) {
		torch::Tensor t = unpackData[i].narrow(0, 0, seqLens[i]);
		unpackVec.push_back(t);
		std::cout << "unpack t " << std::endl << t << std::endl;
	}
	torch::Tensor unpackTensor = torch::cat(unpackVec, 0);
	std::cout << "unpack data " << unpackTensor << std::endl;
}

void testSplit() {
	const int batch = 3;
	std::vector<torch::Tensor> tensorVec;
	std::vector<long> seqLens;
	for (int i = 0; i < batch; i ++) {
		tensorVec.push_back(torch::ones({(batch - i), 2, 2}) * (batch - i));
		seqLens.push_back(batch - i);

		std::cout << "Orig tensor " << i << std::endl << tensorVec[tensorVec.size() - 1] << std::endl;
	}
	std::cout << "seqLens = " << seqLens << std::endl;

	torch::Tensor totalTensor = torch::cat(tensorVec, 0);
	std::cout << "total tensor = " << std::endl << totalTensor << std::endl;

	std::vector<long> indice {0, 1, 2};
	torch::Tensor indiceTensor = torch::from_blob(indice.data(), {indice.size()}, longOpt);
	auto splitTensor = torch::index_select(totalTensor, 0, indiceTensor);
	std::cout << "split tensor " << std::endl << splitTensor << std::endl;
}

void testNarrow() {
	const int batch = 3;
	std::vector<torch::Tensor> tensorVec;
	std::vector<long> seqLens;
	for (int i = 0; i < batch; i ++) {
		tensorVec.push_back(torch::ones({(batch - i), 2, 2}) * (batch - i));
		seqLens.push_back(batch - i);

		std::cout << "Orig tensor " << i << std::endl << tensorVec[tensorVec.size() - 1] << std::endl;
	}
	std::cout << "seqLens = " << seqLens << std::endl;

	torch::Tensor totalTensor = torch::cat(tensorVec, 0);
	std::cout << "total tensor = " << std::endl << totalTensor << std::endl;

	std::vector<torch::Tensor> narrowVec;
	int index = 0;
	for (int i = 0; i < seqLens.size(); i ++) {
		auto narrowTensor = totalTensor.narrow(0, index, seqLens[i]);
		narrowVec.push_back(narrowTensor);
		index += seqLens[i];

		std::cout << "narrow tensor " << std::endl << narrowTensor << std::endl;
	}
}

void testSelect() {
	torch::Tensor orig = torch::rand({2, 3, 4});
	std::cout << "orig = " << std::endl << orig << std::endl;

//	torch::Tensor envH = orig.select(1, 2);
//	std::cout << "envH = " << std::endl << envH << std::endl;

	std::vector<torch::Tensor> hVec;
	for (int i = 0; i < orig.sizes()[1]; i ++) {
		torch::Tensor h = orig.select(1, i);
		std::cout << "h = " << std::endl << h << std::endl;
		hVec.push_back(h);
	}

	torch::Tensor sumH = torch::stack(hVec, 1);
	std::cout << "sumH = " << std::endl << sumH << std::endl;

	std::vector<long> indice{2, 1};
	torch::Tensor indiceTensor = torch::from_blob(indice.data(), {indice.size()}, longOpt);
	torch::Tensor pick = sumH.index_select(1, indiceTensor);
	std::cout << "pick " << std::endl << pick << std::endl;
}

struct episode {
	int seqLen;
	int val;
};
void testSort() {
	const int vecLen = 4;
	std::vector<episode> es;
	for (int i = 0; i < vecLen; i ++) {
		int s = torch::rand({1}).item<float>() * 100;
		int v = s / 10;
		es.push_back({s, v});
		std::cout << "push " << s << ": " << v << std::endl;
	}

	std::sort(es.begin(), es.end(),
			[](const episode& a, const episode& b) -> bool {
				return a.seqLen > b.seqLen;
	});
	for (const auto& e: es) {
		std::cout << "sorted " << e.seqLen << ": " << e.val << std::endl;
	}
}

void testFill() {
	torch::Tensor t = torch::rand({2, 3, 4});
	std::cout << "t = " << std::endl << t << std::endl;

	t[0][1].fill_(0);
	std::cout << "after fill " << std::endl << t << std::endl;
}

void testConstPad() {
	torch::Tensor t0 = torch::ones({5, 2, 2}) * 5;
	torch::Tensor t1 = torch::ones({3, 2, 2}) * 3;
	std::cout << "t0 " << t0 << std::endl;
	std::cout << "t1 " << t1 << std::endl;

	t0 = torch::constant_pad_nd(t0, {0, 4 - 5});
	t1 = torch::constant_pad_nd(t1, {0, 4 - 3});
	std::cout << "-------------> t0 " << t0 << std::endl;
	std::cout << "-------------> t1 " << t1 << std::endl;
}

void testStack() {
	int batch = 5;
	int step = 3;
	std::vector<torch::Tensor> ts;
	for (int i = 0; i < batch; i ++) {
		torch::Tensor t = torch::ones({step, 2, 2}) * i * 10;
		for (int j = 0; j < step; j ++) {
			t[j].add_(j);
		}
		ts.push_back(t);
	}
	torch::Tensor tStack = torch::hstack(ts);
	std::cout << "tStack " << std::endl << tStack << std::endl;
	torch::Tensor tCat = torch::cat(ts, 1);
	std::cout << "tCat " << std::endl << tCat << std::endl;

	torch::Tensor tFlat = tStack.view({batch * step, 2, 2});
	std::cout << "tFlat " << std::endl << tFlat << std::endl;

	std::vector<torch::Tensor> sts = tFlat.split(batch, 0);
	for (auto& t: sts) {
		std::cout << "sts " << std::endl << t << std::endl;
	}
	torch::Tensor input = torch::cat(sts, 1);
	input = input.view({batch, step, 2, 2});
	std::cout << "input " << std::endl << input << std::endl;
}

void testPermute() {
	const int batch = 2;
	const int seq = 3;
	std::vector<float> data(batch * seq, 0);
	for (int i = 0; i < data.size(); i ++) {
		data[i] = i;
	}

	torch::Tensor dt = torch::from_blob(data.data(), {batch, seq, 1});
	std::cout << "dt " << std::endl << dt << std::endl;

	dt = dt.permute({1, 0, 2});
	std::cout << "permute " << std::endl << dt <<std::endl;

	dt = dt.permute({1, 0, 2});
	std::cout << "recover " << std::endl << dt << std::endl;
}
}

int main() {
	testPack();
//	testSplit();
//	testNarrow();
//	testSelect();
//	testSort();
//	testFill();
//	testConstPad();
//	testStack();
//	testPermute();
}

