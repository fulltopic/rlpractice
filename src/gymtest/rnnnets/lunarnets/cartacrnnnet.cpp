/*
 * cartacrnnnet.cpp
 *
 *  Created on: Jan 27, 2022
 *      Author: zf
 */


#include "gymtest/rnnnets/lunarnets/cartacrnnnet.h"

#include <torch/torch.h>

#include <vector>

CartACRNNFcNet::CartACRNNFcNet(int input, int hidden, int output):
	inputNum(input),
	outputNum(output),
	hiddenNum(hidden),
	gru0(torch::nn::GRUOptions(input, hidden).batch_first(true)),
	vOutput(hidden, 1),
	aOutput(hidden, output) {
	register_module("gru0", gru0);
	register_module("vOutput", vOutput);
	register_module("aOutput", aOutput);
}

std::vector<torch::Tensor> CartACRNNFcNet::forward(torch::Tensor input, std::vector<long> seqInput, std::vector<torch::Tensor>& states) {
	//input: {sum(seqLen), others}
	std::vector<torch::Tensor> splitVec;
	int index = 0;
	for (auto seqLen: seqInput) {
		torch::Tensor t = input.narrow(0, index, seqLen);
		index += seqLen;
	}
	auto padSeqTensor = torch::nn::utils::rnn::pad_sequence(splitVec);
	torch::Tensor seqTensor = torch::from_blob(seqInput.data(), {(long)seqInput.size()}, longOpt);
	auto packInput = torch::nn::utils::rnn::pack_padded_sequence(padSeqTensor, seqTensor,  true);


	auto rnnOutput = gru0->forward_with_packed_input(packInput, states[0]);
	states[0] = std::get<1>(rnnOutput);

	//{step, batch, hidden} -> {step * batch, hidden}
	torch::Tensor fcInput = std::get<0>(rnnOutput).data(); //{sum(seqLen), others}

	auto v = vOutput->forward(fcInput);
	auto a = aOutput->forward(fcInput);

	return {a, v};
}

std::vector<torch::Tensor> CartACRNNFcNet::forward(torch::Tensor input, std::vector<torch::Tensor>& states) {
	//input: {batch, others}
	auto rnnOutput = gru0->forward(input, states[0]);
	states[0] = std::get<1>(rnnOutput);

	//{step, batch, hidden} -> {step * batch, hidden}
	torch::Tensor fcInput = std::get<0>(rnnOutput); //{sum(seqLen), others}

	auto v = vOutput->forward(fcInput);
	auto a = aOutput->forward(fcInput);

	return {a, v};
}

