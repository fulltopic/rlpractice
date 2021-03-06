/*
 * cartacgruslimnet.cpp
 *
 *  Created on: Feb 14, 2022
 *      Author: zf
 */



#include "gymtest/rnnnets/lunarnets/cartacgruslim.h"

#include <torch/torch.h>

#include <vector>
#include <iostream>
#include <algorithm>

CartACGRUTruncFcSlimNet::CartACGRUTruncFcSlimNet(int input, int hidden, int output):
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

std::vector<torch::Tensor> CartACGRUTruncFcSlimNet::forward(torch::Tensor input, std::vector<long> seqInput) {
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


//std::vector<torch::Tensor> CartACGRUTruncFcSlimNet::forward(torch::Tensor input, int batchSize, int seqLen, std::vector<torch::Tensor>& states) {
//	std::cout << "Input " << input.sizes() << std::endl;
//	//input: {batch * seqLen , others}
//	torch::Tensor output = input.view({batchSize, seqLen, -1});
//	std::cout << "gru input " << output.sizes() << std::endl;
//
//	auto rnnOutput = gru0->forward(output, states[0]);
//
//	states[0] = std::get<1>(rnnOutput);
//
//	auto splitVec = std::get<0>(rnnOutput).split(batchSize, 0);
//	torch::Tensor fcInput = torch::cat(splitVec, 1).view({batchSize * seqLen, -1});
//	//{step, batch, hidden} -> {step * batch, hidden}
//	std::cout << "fcInput batch ------------------------------->" << fcInput.sizes() << std::endl;
//
//	auto v = vOutput->forward(fcInput);
//	auto a = aOutput->forward(fcInput);
//
//	return {a, v};
//}

std::vector<torch::Tensor> CartACGRUTruncFcSlimNet::forward(torch::Tensor input, int batchSize, int seqLen, std::vector<torch::Tensor>& states, torch::Device deviceType) {
//	std::cout << "Input " << input.sizes() << std::endl;
	//input: {batch * seqLen , others}
	std::vector<long> indice(batchSize * seqLen, 0);
	for (int i = 0; i < batchSize * seqLen; i ++) {
		indice[i] = i;
	}
	std::random_shuffle(indice.begin(), indice.end());
	std::vector<long> rIndice(indice.size(), 0);
	for (int i = 0; i < indice.size(); i ++) {
		rIndice[indice[i]] = i;
	}
	torch::Tensor indiceTensor = torch::from_blob(indice.data(), {batchSize * seqLen}, longOpt).to(deviceType);
	torch::Tensor rIndiceTensor = torch::from_blob(rIndice.data(), {batchSize * seqLen}, longOpt).to(deviceType);

	torch::Tensor output = input.view({batchSize, seqLen, -1});
//	std::cout << "gru input " << output.sizes() << std::endl;

	auto rnnOutput = gru0->forward(output, states[0]); //{batch, seq, others}

	states[0] = std::get<1>(rnnOutput);

//	std::cout << "rnn output " << std::get<0>(rnnOutput).sizes() << std::endl;
//	auto splitVec = std::get<0>(rnnOutput).split(batchSize, 0);
//	torch::Tensor fcInput = torch::cat(splitVec, 1).view({batchSize * seqLen, -1});
	torch::Tensor fcInput = std::get<0>(rnnOutput).contiguous().view({batchSize * seqLen, -1});
	fcInput = fcInput.index_select(0, indiceTensor);
	//{step, batch, hidden} -> {step * batch, hidden}
//	std::cout << "fcInput batch ------------------------------->" << fcInput.sizes() << std::endl;

	auto v = vOutput->forward(fcInput);
	auto a = aOutput->forward(fcInput);

	v = v.index_select(0, rIndiceTensor);
	a = a.index_select(0, rIndiceTensor);

	return {a, v};
}

std::vector<torch::Tensor> CartACGRUTruncFcSlimNet::forwardNext(torch::Tensor input, int batchSize, int seqLen, std::vector<torch::Tensor> states, torch::Device deviceType) {
//	std::cout << "Input " << input.sizes() << std::endl;
	//input: {batch * seqLen , others}
	std::vector<long> indice(batchSize * seqLen, 0);
	for (int i = 0; i < batchSize * seqLen; i ++) {
		indice[i] = i;
	}
	std::random_shuffle(indice.begin(), indice.end());
	std::vector<long> rIndice(indice.size(), 0);
	for (int i = 0; i < indice.size(); i ++) {
		rIndice[indice[i]] = i;
	}
	torch::Tensor indiceTensor = torch::from_blob(indice.data(), {batchSize * seqLen}, longOpt).to(deviceType);
	torch::Tensor rIndiceTensor = torch::from_blob(rIndice.data(), {batchSize * seqLen}, longOpt).to(deviceType);

	torch::Tensor output = input.view({batchSize, seqLen, -1});
//	std::cout << "gru input " << output.sizes() << std::endl;

	auto rnnOutput = gru0->forward(output, states[0]); //{batch, seq, others}


//	std::cout << "rnn output " << std::get<0>(rnnOutput).sizes() << std::endl;
//	auto splitVec = std::get<0>(rnnOutput).split(batchSize, 0);
//	torch::Tensor fcInput = torch::cat(splitVec, 1).view({batchSize * seqLen, -1});
	torch::Tensor fcInput = std::get<0>(rnnOutput).contiguous().view({batchSize * seqLen, -1});
//	fcInput = fcInput.index_select(0, indiceTensor);
	//{step, batch, hidden} -> {step * batch, hidden}
//	std::cout << "fcInput batch ------------------------------->" << fcInput.sizes() << std::endl;

	auto v = vOutput->forward(fcInput);
	auto a = aOutput->forward(fcInput);

//	v = v.index_select(0, rIndiceTensor);
//	a = a.index_select(0, rIndiceTensor);

	return {a, v};
}


std::vector<torch::Tensor> CartACGRUTruncFcSlimNet::forward(torch::Tensor input, std::vector<torch::Tensor>& states) {
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

std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> CartACGRUTruncFcSlimNet::forwardNext(torch::Tensor input, std::vector<torch::Tensor> states) {
	//input: {batch, others}
	auto rnnOutput = gru0->forward(input, states[0]);
	auto nextState = std::get<1>(rnnOutput);

	//{step, batch, hidden} -> {step * batch, hidden}
//	std::cout << "fcInput step " << std::get<0>(rnnOutput).sizes() << std::endl;
	torch::Tensor fcInput = std::get<0>(rnnOutput).squeeze(1); //{batch, others}


	auto v = vOutput->forward(fcInput);
	auto a = aOutput->forward(fcInput);

	return {{a, v}, {nextState}};
}


std::vector<torch::Tensor> CartACGRUTruncFcSlimNet::createHStates(const int envNum, torch::Device deviceType) {
	torch::Tensor hState = torch::zeros({1, envNum, hiddenNum}).to(deviceType); //layernum, envNum, hiddenNum

	return {hState};
}

void CartACGRUTruncFcSlimNet::resetHState(const int index, std::vector<torch::Tensor>& states) {
	states[0][0][index].fill_(0);
}

//Return independent tensor
std::vector<torch::Tensor> CartACGRUTruncFcSlimNet::getHState(const int envIndex, std::vector<torch::Tensor>& states) {
	torch::Tensor envHState = states[0].select(1, envIndex).clone().detach();

	return {envHState};
}

