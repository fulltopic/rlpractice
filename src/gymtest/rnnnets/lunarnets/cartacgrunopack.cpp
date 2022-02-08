/*
 * cartacgrunopack.cpp
 *
 *  Created on: Feb 4, 2022
 *      Author: zf
 */




/*
 * cartacrnnnet.cpp
 *
 *  Created on: Jan 27, 2022
 *      Author: zf
 */


#include "gymtest/rnnnets/lunarnets/cartacgrunopack.h"

#include <torch/torch.h>

#include <vector>
#include <iostream>

CartACRNNNoPackNet::CartACRNNNoPackNet(int input, int hidden, int output):
	inputNum(input),
	outputNum(output),
	hiddenNum(hidden),
	fc0(input, hidden),
	gru1(torch::nn::GRUOptions(hidden, hidden).batch_first(true)),
	vOutput(hidden, 1),
	aOutput(hidden, output) {
	register_module("fc0", fc0);
	register_module("gru1", gru1);
	register_module("vOutput", vOutput);
	register_module("aOutput", aOutput);
}

std::vector<torch::Tensor> CartACRNNNoPackNet::forwardTrain(torch::Tensor input) {
	//input: {batch, others}
	auto fcOutput = fc0->forward(input);
	fcOutput = torch::relu(fcOutput);

	auto rnnOutput = gru1->forward(fcOutput);
//	states[0] = std::get<1>(rnnOutput);

	//{step, batch, hidden} -> {step * batch, hidden}
//	std::cout << "fcInput step " << std::get<0>(rnnOutput).sizes() << std::endl;
	torch::Tensor fcInput = std::get<0>(rnnOutput); //{batch, step, others}


	auto v = vOutput->forward(fcInput);
	auto a = aOutput->forward(fcInput);

	return {a, v};
}



std::vector<torch::Tensor> CartACRNNNoPackNet::forward(torch::Tensor input, std::vector<torch::Tensor>& states) {
	//input: {batch, others}
	auto fcOutput = fc0->forward(input);
	fcOutput = torch::relu(fcOutput);

	auto rnnOutput = gru1->forward(fcOutput, states[0]);
	states[0] = std::get<1>(rnnOutput);

	//{step, batch, hidden} -> {step * batch, hidden}
//	std::cout << "fcInput step " << std::get<0>(rnnOutput).sizes() << std::endl;
	torch::Tensor fcInput = std::get<0>(rnnOutput).squeeze(1); //{batch, others}


	auto v = vOutput->forward(fcInput);
	auto a = aOutput->forward(fcInput);

	return {a, v};
}

std::vector<torch::Tensor> CartACRNNNoPackNet::forward(torch::Tensor input, std::vector<long> seqInput) {
	//input: {sum(seqLen), others}
	auto fcOutput = fc0->forward(input);
	fcOutput = torch::relu(fcOutput);

	std::vector<torch::Tensor> splitVec;
	int index = 0;
	for (auto seqLen: seqInput) {
//		std::cout << "index = " << index << " seq = " << seqLen << std::endl;
		torch::Tensor t = fcOutput.narrow(0, index, seqLen);
		splitVec.push_back(t);
		index += seqLen;
	}
	auto padSeqTensor = torch::nn::utils::rnn::pad_sequence(splitVec, true);
//	std::cout << "padSeqTensor " << padSeqTensor.sizes() << std::endl;
	torch::Tensor seqTensor = torch::from_blob(seqInput.data(), {(long)seqInput.size()}, longOpt); //no to(device)
	auto packInput = torch::nn::utils::rnn::pack_padded_sequence(padSeqTensor, seqTensor,  true);


	auto rnnOutput = gru1->forward_with_packed_input(packInput);

	auto unpack = torch::nn::utils::rnn::pad_packed_sequence(std::get<0>(rnnOutput), true);
	auto unpackData = std::get<0>(unpack);
	std::vector<torch::Tensor> unpackVec;
	for (int i = 0; i < seqInput.size(); i ++) {
		torch::Tensor t = unpackData[i].narrow(0, 0, seqInput[i]);
//		std::cout << "unpack data " << t.sizes() << std::endl;
		unpackVec.push_back(t);
	}
	auto fcInput = torch::cat(unpackVec, 0);
//	states[0] = std::get<1>(rnnOutput);

	//{step, batch, hidden} -> {step * batch, hidden}
//	torch::Tensor fcInput = std::get<0>(rnnOutput).data(); //{sum(seqLen), others}
//	std::cout << "fcInput batch ------------------------------->" << fcInput.sizes() << std::endl;

	auto v = vOutput->forward(fcInput);
	auto a = aOutput->forward(fcInput);

	return {a, v};
}

std::vector<torch::Tensor> CartACRNNNoPackNet::createHStates(const int envNum, torch::Device deviceType) {
	torch::Tensor hState = torch::zeros({1, envNum, hiddenNum}).to(deviceType); //layernum, envNum, hiddenNum

	return {hState};
}
