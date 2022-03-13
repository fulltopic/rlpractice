/*
 * airacgrunordnet.cpp
 *
 *  Created on: Mar 7, 2022
 *      Author: zf
 */




#include "gymtest/rnnnets/airnets/airacgrunordnet.h"
#include "gymtest/utils/netinitutils.h"

#include <torch/torch.h>

AirACHONRGRUNet::AirACHONRGRUNet(int aNum, int hidden):
	conv0(torch::nn::Conv2dOptions(4, 32, 8).stride(4)),
	conv1(torch::nn::Conv2dOptions(32, 64, 4).stride(2)),
	conv2(torch::nn::Conv2dOptions(64, 64, 3).stride(1)),
	gru0(torch::nn::GRUOptions(3136, hidden).batch_first(true)),
//	afc(3136, 512),
//	vfc(3136, 512),
	aOut(hidden, aNum),
	vOut(hidden, 1),
	actionNum(aNum),
	hiddenNum(hidden)
//	fc3(1568, 18)
{
	register_module("conv0", conv0);
	register_module("conv1", conv1);
	register_module("conv2", conv2);
	register_module("gru0", gru0);
//	register_module("afc", afc);
//	register_module("vfc", vfc);
	register_module("aOut", aOut);
	register_module("vOut", vOut);

	//TODO: init gru0
	NetInitUtils::Init_weights(conv0->named_parameters(), sqrt(2.0), 0, NetInitUtils::Kaiming);
	NetInitUtils::Init_weights(conv1->named_parameters(), sqrt(2.0), 0, NetInitUtils::Kaiming);
	NetInitUtils::Init_weights(conv2->named_parameters(), sqrt(2.0), 0, NetInitUtils::Kaiming);
//	NetInitUtils::Init_weights(afc->named_parameters(), sqrt(2.0), 0, NetInitUtils::Xavier);
//	NetInitUtils::Init_weights(vfc->named_parameters(), sqrt(2.0), 0, NetInitUtils::Xavier);
	NetInitUtils::Init_weights(aOut->named_parameters(), sqrt(2.0), 0, NetInitUtils::Xavier);
	NetInitUtils::Init_weights(vOut->named_parameters(), sqrt(2.0), 0, NetInitUtils::Xavier);

	NetInitUtils::Init_weights(gru0->named_parameters(), sqrt(1.0), -1, NetInitUtils::Xavier);
}

std::vector<torch::Tensor> AirACHONRGRUNet::forward(torch::Tensor input, std::vector<torch::Tensor>& states) {
//	input: {batch, 1, others}
//	std::cout << "single input " << input.sizes() << std::endl;
	input = input.squeeze(1);
//	std::cout << "single input squeeze " << input.sizes() << std::endl;
	torch::Tensor output = conv0->forward(input);
	output = torch::relu(output);

	output = conv1->forward(output);
	output = torch::relu(output);

	output = conv2->forward(output);
	output = torch::relu(output);

	//single step
//	std::cout << "conv output " << output.sizes() << std::endl;
	output = output.view({input.sizes()[0], 1, -1});
//	std::cout << "conv output view " << output.sizes() << std::endl;

//	std::cout << "gru state " << states[0].sizes() << std::endl;
	auto gruOutput = gru0->forward(output, states[0]);
	states[0] = std::get<1>(gruOutput);
//	std::cout << "gru new state " << states[0].sizes() << std::endl;

//	std::cout << "fcInput " << std::get<0>(gruOutput).sizes() << std::endl;
	auto fcInput = std::get<0>(gruOutput).squeeze(1);
//	std::cout << "fcInput squeeze" << fcInput.sizes() << std::endl;

//	auto aOutput = afc->forward(fcInput);
//	auto vOutput = vfc->forward(fcInput);
//	aOutput = torch::relu(aOutput);
//	vOutput = torch::relu(vOutput);

//	aOutput = aOut->forward(aOutput);
//	vOutput = vOut->forward(vOutput);

	auto aOutput = aOut->forward(fcInput);
	auto vOutput = vOut->forward(fcInput);
//	std::cout << "aOutput " << aOutput.sizes() << std::endl;
//	std::cout << "vOutput " << vOutput.sizes() << std::endl;

	return {aOutput, vOutput};
}

std::vector<torch::Tensor> AirACHONRGRUNet::forward(torch::Tensor input, std::vector<long> seqInput, std::vector<torch::Tensor> states) {
	//input {sum(seqLen), others}
//	std::cout << "--------------------------------> batch input " << input.sizes() << std::endl;
	const int batchSize = seqInput.size();
	int seqSum = 0;
	for (const auto& seq: seqInput) {
		seqSum += seq;
	}
	std::vector<long> indice(seqSum, 0);
	for (int i = 0; i < seqSum; i ++) {
		indice[i] = i;
	}
	std::random_shuffle(indice.begin(), indice.end());
	std::vector<long> rIndice(indice.size(), 0);
	for (int i = 0; i < indice.size(); i ++) {
		rIndice[indice[i]] = i;
	}
	torch::Tensor indiceTensor = torch::from_blob(indice.data(), {seqSum}, longOpt).to(input.device());
	torch::Tensor rIndiceTensor = torch::from_blob(rIndice.data(), {seqSum}, longOpt).to(input.device());

	torch::Tensor convInput = input.index_select(0, indiceTensor);

	torch::Tensor output = conv0->forward(convInput);
	output = torch::relu(output);

	output = conv1->forward(output);
	output = torch::relu(output);

	output = conv2->forward(output);
	output = torch::relu(output);

	//{sum(seqLen), mul(others)}
//	std::cout << "conv output " << output.sizes() << std::endl;
	output = output.index_select(0, rIndiceTensor);
	output = output.view({input.sizes()[0], -1});
//	std::cout << "conv output view " << output.sizes() << std::endl;

	std::vector<torch::Tensor> splitVec;
	int index = 0;
	for (auto seqLen: seqInput) {
//		std::cout << "index = " << index << " seq = " << seqLen << std::endl;
		torch::Tensor t = output.narrow(0, index, seqLen);
		splitVec.push_back(t);
		index += seqLen;
//		std::cout << "t = " << t.sizes() << std::endl;
	}
	auto padSeqTensor = torch::nn::utils::rnn::pad_sequence(splitVec, true);
//	std::cout << "padSeq " << padSeqTensor.sizes() << std::endl;
	torch::Tensor seqTensor = torch::from_blob(seqInput.data(), {(long)seqInput.size()}, longOpt);
	auto packInput = torch::nn::utils::rnn::pack_padded_sequence(padSeqTensor, seqTensor,  true);

	auto rnnOutput = gru0->forward_with_packed_input(packInput, states[0]);
	states[0] = std::get<1>(rnnOutput);
//	std::cout << "gru new state " << states[0].sizes() << std::endl;


	auto unpack = torch::nn::utils::rnn::pad_packed_sequence(std::get<0>(rnnOutput), true);
	auto unpackData = std::get<0>(unpack);
//	std::cout << "unpack " << unpackData.sizes() << std::endl;
	std::vector<torch::Tensor> unpackVec;
	for (int i = 0; i < seqInput.size(); i ++) {
		torch::Tensor t = unpackData[i].narrow(0, 0, seqInput[i]);
//		std::cout << "unpack data " << t.sizes() << std::endl;
		unpackVec.push_back(t);
	}
	auto fcInput = torch::cat(unpackVec, 0); //{sum(seqLen), mul(others)}
	fcInput = fcInput.index_select(0, indiceTensor);
//	std::cout << "fcInput " << fcInput.sizes() << std::endl;

//	auto aOutput = afc->forward(fcInput);
//	auto vOutput = vfc->forward(fcInput);
//	aOutput = torch::relu(aOutput);
//	vOutput = torch::relu(vOutput);
//
//	aOutput = aOut->forward(aOutput);
//	vOutput = vOut->forward(vOutput);

	auto aOutput = aOut->forward(fcInput);
	auto vOutput = vOut->forward(fcInput);
	aOutput = aOutput.index_select(0, rIndiceTensor);
	vOutput = vOutput.index_select(0, rIndiceTensor);
//	std::cout << "aOutput " << aOutput.sizes() << std::endl;
//	std::cout << "vOutput " << vOutput.sizes() << std::endl;

	return {aOutput, vOutput};
}

std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> AirACHONRGRUNet::forwardNext(
		torch::Tensor input, std::vector<torch::Tensor> states) {
	//	input: {batch, 1, others}
	//	std::cout << "single input " << input.sizes() << std::endl;
		input = input.squeeze(1);
	//	std::cout << "single input squeeze " << input.sizes() << std::endl;
		torch::Tensor output = conv0->forward(input);
		output = torch::relu(output);

		output = conv1->forward(output);
		output = torch::relu(output);

		output = conv2->forward(output);
		output = torch::relu(output);

		//single step
	//	std::cout << "conv output " << output.sizes() << std::endl;
		output = output.view({input.sizes()[0], 1, -1});
	//	std::cout << "conv output view " << output.sizes() << std::endl;

	//	std::cout << "gru state " << states[0].sizes() << std::endl;
		auto gruOutput = gru0->forward(output, states[0]);
		auto nextStepState = std::get<1>(gruOutput);
	//	std::cout << "gru new state " << states[0].sizes() << std::endl;

	//	std::cout << "fcInput " << std::get<0>(gruOutput).sizes() << std::endl;
		auto fcInput = std::get<0>(gruOutput).squeeze(1);
	//	std::cout << "fcInput squeeze" << fcInput.sizes() << std::endl;

	//	auto aOutput = afc->forward(fcInput);
	//	auto vOutput = vfc->forward(fcInput);
	//	aOutput = torch::relu(aOutput);
	//	vOutput = torch::relu(vOutput);

	//	aOutput = aOut->forward(aOutput);
	//	vOutput = vOut->forward(vOutput);

		auto aOutput = aOut->forward(fcInput);
		auto vOutput = vOut->forward(fcInput);
	//	std::cout << "aOutput " << aOutput.sizes() << std::endl;
	//	std::cout << "vOutput " << vOutput.sizes() << std::endl;

		return {{aOutput, vOutput}, {nextStepState}};
}

std::vector<torch::Tensor> AirACHONRGRUNet::forwardNext(
		torch::Tensor input, int batchSize, std::vector<long> seqInput, std::vector<torch::Tensor> states, torch::Device deviceType) {
	//input {sum(seqLen), others}
//	std::cout << "--------------------------------> batch input " << input.sizes() << std::endl;
//	const int batchSize = seqInput.size();
	int seqSum = 0;
	for (const auto& seq: seqInput) {
		seqSum += seq;
	}

//	std::vector<long> indice(seqSum, 0);
//	for (int i = 0; i < seqSum; i ++) {
//		indice[i] = i;
//	}
//	std::random_shuffle(indice.begin(), indice.end());
//	std::vector<long> rIndice(indice.size(), 0);
//	for (int i = 0; i < indice.size(); i ++) {
//		rIndice[indice[i]] = i;
//	}
//	torch::Tensor indiceTensor = torch::from_blob(indice.data(), {seqSum}, longOpt).to(input.device());
//	torch::Tensor rIndiceTensor = torch::from_blob(rIndice.data(), {seqSum}, longOpt).to(input.device());
//
//	torch::Tensor convInput = input.index_select(0, indiceTensor);

	torch::Tensor convInput = input;
	torch::Tensor output = conv0->forward(convInput);
	output = torch::relu(output);

	output = conv1->forward(output);
	output = torch::relu(output);

	output = conv2->forward(output);
	output = torch::relu(output);

	//{sum(seqLen), mul(others)}
//	std::cout << "conv output " << output.sizes() << std::endl;
//	output = output.index_select(0, rIndiceTensor);
	output = output.view({input.sizes()[0], -1});
//	std::cout << "conv output view " << output.sizes() << std::endl;

	std::vector<torch::Tensor> splitVec;
	int index = 0;
	for (auto seqLen: seqInput) {
//		std::cout << "index = " << index << " seq = " << seqLen << std::endl;
		torch::Tensor t = output.narrow(0, index, seqLen);
		splitVec.push_back(t);
		index += seqLen;
//		std::cout << "t = " << t.sizes() << std::endl;
	}
	auto padSeqTensor = torch::nn::utils::rnn::pad_sequence(splitVec, true);
//	std::cout << "padSeq " << padSeqTensor.sizes() << std::endl;
	torch::Tensor seqTensor = torch::from_blob(seqInput.data(), {(long)seqInput.size()}, longOpt);
	auto packInput = torch::nn::utils::rnn::pack_padded_sequence(padSeqTensor, seqTensor,  true);

	auto rnnOutput = gru0->forward_with_packed_input(packInput, states[0]);
//	states[0] = std::get<1>(rnnOutput);
//	std::cout << "gru new state " << states[0].sizes() << std::endl;


	auto unpack = torch::nn::utils::rnn::pad_packed_sequence(std::get<0>(rnnOutput), true);
	auto unpackData = std::get<0>(unpack);
//	std::cout << "unpack " << unpackData.sizes() << std::endl;
	std::vector<torch::Tensor> unpackVec;
	for (int i = 0; i < seqInput.size(); i ++) {
		torch::Tensor t = unpackData[i].narrow(0, 0, seqInput[i]);
//		std::cout << "unpack data " << t.sizes() << std::endl;
		unpackVec.push_back(t);
	}
	auto fcInput = torch::cat(unpackVec, 0); //{sum(seqLen), mul(others)}
//	fcInput = fcInput.index_select(0, indiceTensor);
//	std::cout << "fcInput " << fcInput.sizes() << std::endl;

//	auto aOutput = afc->forward(fcInput);
//	auto vOutput = vfc->forward(fcInput);
//	aOutput = torch::relu(aOutput);
//	vOutput = torch::relu(vOutput);
//
//	aOutput = aOut->forward(aOutput);
//	vOutput = vOut->forward(vOutput);

	auto aOutput = aOut->forward(fcInput);
	auto vOutput = vOut->forward(fcInput);
//	aOutput = aOutput.index_select(0, rIndiceTensor);
//	vOutput = vOutput.index_select(0, rIndiceTensor);
//	std::cout << "aOutput " << aOutput.sizes() << std::endl;
//	std::cout << "vOutput " << vOutput.sizes() << std::endl;

	return {aOutput, vOutput};
}


std::vector<torch::Tensor> AirACHONRGRUNet::createHStates(const int envNum, torch::Device deviceType) {
	torch::Tensor hState = torch::zeros({1, envNum, hiddenNum}).to(deviceType); //layernum, envNum, hiddenNum

	return {hState};
}

void AirACHONRGRUNet::resetHState(const int index, std::vector<torch::Tensor>& states) {
	states[0][0][index].fill_(0);
}

//Return independent tensor
//states: cellNum * {layerNum, batchNum, hiddenNum}
std::vector<torch::Tensor> AirACHONRGRUNet::getHState(const int envIndex, std::vector<torch::Tensor>& states) {
	torch::Tensor envHState = states[0].select(1, envIndex).clone().detach();

	return {envHState};
}


