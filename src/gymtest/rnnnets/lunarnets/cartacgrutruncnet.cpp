/*
 * cartacgrutruncnet.cpp
 *
 *  Created on: Feb 7, 2022
 *      Author: zf
 */



#include "gymtest/rnnnets/lunarnets/cartacgrutruncnet.h"

#include <torch/torch.h>

#include <vector>
#include <iostream>

CartACGRUTruncFcNet::CartACGRUTruncFcNet(int input, int hidden, int output):
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

std::vector<torch::Tensor> CartACGRUTruncFcNet::forward(torch::Tensor input, std::vector<long> seqInput) {
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
//	std::cout << "padSeqTensor " << padSeqTensor << std::endl;
	torch::Tensor seqTensor = torch::from_blob(seqInput.data(), {(long)seqInput.size()}, longOpt);
	auto packInput = torch::nn::utils::rnn::pack_padded_sequence(padSeqTensor, seqTensor,  true);
//	std::cout << "packInput data " << packInput.data() << std::endl;
//	std::cout << "packInput batch size " << packInput.batch_sizes() << std::endl;
//	std::cout << "packInput sorted " << packInput.sorted_indices() << std::endl;
//	std::cout << "packInput unsorted " << packInput.unsorted_indices() << std::endl;


	auto rnnOutput = gru0->forward_with_packed_input(packInput);

	auto unpack = torch::nn::utils::rnn::pad_packed_sequence(std::get<0>(rnnOutput), true);
	auto unpackData = std::get<0>(unpack);
	std::vector<torch::Tensor> unpackVec;
	for (int i = 0; i < seqInput.size(); i ++) {
		torch::Tensor t = unpackData[i].narrow(0, 0, seqInput[i]);
//		std::cout << "unpack data " << t.sizes() << std::endl;
		unpackVec.push_back(t);
	}
	auto fcInput = torch::cat(unpackVec, 0);
//	std::cout << "fcInput " << fcInput.sizes() << std::endl;
//	std::cout << "rnnOutput " << std::get<1>(rnnOutput) << std::endl;
//	states[0] = std::get<1>(rnnOutput);

	//{step, batch, hidden} -> {step * batch, hidden}
//	torch::Tensor fcInput = std::get<0>(rnnOutput).data(); //{sum(seqLen), others}
//	std::cout << "fcInput batch " << fcInput << std::endl;

	auto v = vOutput->forward(fcInput);
	auto a = aOutput->forward(fcInput);

	return {a, v};
}


std::vector<torch::Tensor> CartACGRUTruncFcNet::forward(torch::Tensor input, std::vector<long> seqInput, std::vector<torch::Tensor>& states) {
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


	auto rnnOutput = gru0->forward_with_packed_input(packInput, states[0]);
	states[0] = std::get<1>(rnnOutput);

	auto unpack = torch::nn::utils::rnn::pad_packed_sequence(std::get<0>(rnnOutput), true);
	auto unpackData = std::get<0>(unpack);
	std::vector<torch::Tensor> unpackVec;
	for (int i = 0; i < seqInput.size(); i ++) {
		torch::Tensor t = unpackData[i].narrow(0, 0, seqInput[i]);
//		std::cout << "unpack data " << t.sizes() << std::endl;
		unpackVec.push_back(t);
	}
	//{step, batch, hidden} -> {step * batch, hidden}
	auto fcInput = torch::cat(unpackVec, 0);
//	std::cout << "fcInput batch ------------------------------->" << fcInput.sizes() << std::endl;

	auto v = vOutput->forward(fcInput);
	auto a = aOutput->forward(fcInput);

	return {a, v};
}



std::vector<torch::Tensor> CartACGRUTruncFcNet::forward(torch::Tensor input, std::vector<torch::Tensor>& states) {
	//input: {batch, others}
	auto rnnOutput = gru0->forward(input, states[0]);
	states[0] = std::get<1>(rnnOutput);

	//{step, batch, hidden} -> {step * batch, hidden}
//	std::cout << "fcInput step " << std::get<0>(rnnOutput).sizes() << std::endl;
	torch::Tensor fcInput = std::get<0>(rnnOutput).squeeze(1); //{batch, others}


	auto v = vOutput->forward(fcInput);
	auto a = aOutput->forward(fcInput);

	return {a, v};
}

std::vector<torch::Tensor> CartACGRUTruncFcNet::createHStates(const int envNum, torch::Device deviceType) {
	torch::Tensor hState = torch::zeros({1, envNum, hiddenNum}).to(deviceType); //layernum, envNum, hiddenNum

	return {hState};
}

void CartACGRUTruncFcNet::resetHState(const int index, std::vector<torch::Tensor>& states) {
	states[0][0][index].fill_(0);
}
