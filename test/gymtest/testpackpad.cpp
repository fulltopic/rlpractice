/*
 * testpackpad.cpp
 *
 *  Created on: Jan 29, 2022
 *      Author: zf
 */


#include <iostream>
#include <vector>

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
}

int main() {
	testPack();
//	testSplit();
//	testNarrow();
}

