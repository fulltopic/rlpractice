/*
 * cartaclstmnet.cpp
 *
 *  Created on: Feb 3, 2022
 *      Author: zf
 */


#include "gymtest/rnnnets/lunarnets/cartaclstmnet.h"

#include <torch/torch.h>

#include <vector>
#include <iostream>

CartACLSTMFcNet::CartACLSTMFcNet(int input, int hidden, int output):
	inputNum(input),
	outputNum(output),
	hiddenNum(hidden),
	lstm0(torch::nn::LSTMOptions(input, hidden).batch_first(true)),
	vOutput(hidden, 1),
	aOutput(hidden, output) {
	register_module("lstm0", lstm0);
	register_module("vOutput", vOutput);
	register_module("aOutput", aOutput);
}

std::vector<torch::Tensor> CartACLSTMFcNet::forward(torch::Tensor input, std::vector<long> seqInput) {
//	std::cout << "Input " << input.sizes() << std::endl;
	//input: {sum(seqLen), others}
	std::vector<torch::Tensor> splitVec;
	int index = 0;
	for (auto seqLen: seqInput) {
//		std::cout << "index = " << index << " seq = " << seqLen << std::endl;
		torch::Tensor t = input.narrow(0, index, seqLen);
		splitVec.push_back(t);
		index += seqLen;
	}
	auto padSeqTensor = torch::nn::utils::rnn::pad_sequence(splitVec, true);
//	std::cout << "padSeqTensor " << padSeqTensor.sizes() << std::endl;
	torch::Tensor seqTensor = torch::from_blob(seqInput.data(), {(long)seqInput.size()}, longOpt);
	auto packInput = torch::nn::utils::rnn::pack_padded_sequence(padSeqTensor, seqTensor,  true);


	auto rnnOutput = lstm0->forward_with_packed_input(packInput);
//	states[0] = std::get<1>(rnnOutput);

	//{step, batch, hidden} -> {step * batch, hidden}
	torch::Tensor fcInput = std::get<0>(rnnOutput).data(); //{sum(seqLen), others}

	auto v = vOutput->forward(fcInput);
	auto a = aOutput->forward(fcInput);

	return {a, v};
}

std::vector<torch::Tensor> CartACLSTMFcNet::forward(torch::Tensor input, std::vector<torch::Tensor>& cx, std::vector<torch::Tensor>& hx) {
	//input: {batch, others}
	auto rnnOutput = lstm0->forward(input, std::make_tuple(cx[0], hx[0]));
	auto states = std::get<1>(rnnOutput);
	cx[0] = std::get<0>(states);
	hx[0] = std::get<1>(states);

	//{step, batch, hidden} -> {step * batch, hidden}
	torch::Tensor fcInput = std::get<0>(rnnOutput).squeeze(1); //{batch, others}

	auto v = vOutput->forward(fcInput);
	auto a = aOutput->forward(fcInput);

	return {a, v};
}

