/*
 * a3ctcpclienthanle.hpp
 *
 *  Created on: Nov 10, 2021
 *      Author: zf
 */

#ifndef INC_A3C_A3CTCPCLIENTHANLE_HPP_
#define INC_A3C_A3CTCPCLIENTHANLE_HPP_

//#define BOOST_ASIO_ENABLE_HANDLER_TRACKING
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/algorithm/string.hpp>

#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <thread>
#include <mutex>

#include <torch/torch.h>

#include "a3c/a3ctcpconfig.h"
#include "a3c/a3ctcpclientconn.h"
#include "a3c/a3ctcpmsghds.h"
#include "a3c/dummyfuncs.h"

template<typename NetType>
class A3CTCPClientHandle : public std::enable_shared_from_this<A3CTCPClientHandle<NetType>> {
private:
	A3CTCPClientHandle(boost::asio::io_service& iio, NetType& iNet);

	NetType& net;
	std::shared_ptr<A3CTCPClientConn> conn;
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("a3cclienthandle");

	std::vector<torch::Tensor> grads;
	std::vector<torch::Tensor> targetParams;
	boost::array<char, A3CTCPConfig::BufCap> targetSndBuf; //TODO: buffer size to be extended
	boost::array<char, A3CTCPConfig::BufCap> gradSndBuf; //TODO: buffer size to be extended


	std::ostringstream gradStream;
	std::string gradStr;
	uint64_t sendGradIndex = 0;

	std::stringstream targetStream;
	uint64_t getTargetIndex = 0;

	//TODO: Only one sending at one moment
	std::mutex targetMutex;
	std::condition_variable targetCond;
	volatile bool targetUpdating = false;

public:
	~A3CTCPClientHandle();
	A3CTCPClientHandle(const A3CTCPClientHandle&) = delete;

	void start();

	static std::shared_ptr<A3CTCPClientHandle<NetType>> Create(boost::asio::io_service& iio, NetType& iNet);

	void addGrad(std::vector<torch::Tensor>& delta);
	void sendGrad();
	void sendingGrad(void* dataPtr, size_t len);
	void completeGrad();

	void startUpdateTarget();
	void updateTarget(void* data, std::size_t len);
	void completeTarget();
	void syncTarget();

	void sendTest();
	void rcvTest(void* data, std::size_t len);

	void processRcv(void* dataPtr, size_t len);
};

template<typename NetType>
A3CTCPClientHandle<NetType>::A3CTCPClientHandle(boost::asio::io_service& iio, NetType& iNet):
	net(iNet)
{
	conn = A3CTCPClientConn::Create(iio);

	auto params = net.parameters();
	for (const auto& param: params) {
		auto shape = param.sizes();
		targetParams.push_back(torch::zeros(shape));
	}

	//gradds set in add
}
 template<typename NetType>
 A3CTCPClientHandle<NetType>::~A3CTCPClientHandle() {
	 //TODO:
 }

 template<typename NetType>
 std::shared_ptr<A3CTCPClientHandle<NetType>> A3CTCPClientHandle<NetType>::Create(boost::asio::io_service& iio, NetType& iNet) {
	 return std::shared_ptr<A3CTCPClientHandle<NetType>>(new A3CTCPClientHandle<NetType>(iio, iNet));
 }

 template<typename NetType>
 void A3CTCPClientHandle<NetType>::start() {
	 std::function<void(void*, std::size_t len)> func = std::bind(&A3CTCPClientHandle<NetType>::processRcv, this,
			 std::placeholders::_1,
			 std::placeholders::_2);


	 conn->setRcvFunc(func);
	 conn->start();
 }

template<typename NetType>
void  A3CTCPClientHandle<NetType>::addGrad(std::vector<torch::Tensor>& delta) {
//	LOG4CXX_INFO(logger, "add delta to grad sum");
	 if (grads.size() == 0) {
		 for (const auto& d: delta) {
			 grads.push_back(torch::zeros(d.sizes()));
		 }
	 }

	 for (int i = 0; i < grads.size(); i ++) {
		 if (delta[i].numel() > 0) {
			 grads[i].add_(delta[i].to(torch::kCPU));
		 }
	 }

//	 LOG4CXX_INFO(logger, "grad updated");
}

template<typename NetType>
void A3CTCPClientHandle<NetType>::sendGrad() {
	gradStream.clear();
	gradStream.str("");
	torch::save(grads, gradStream);
	gradStr = gradStream.str();
	sendGradIndex = 0;


	GradSyncReq* req = A3CTCPCmdFactory::CreateGradSyncReq(gradSndBuf.data());
//	uint64_t cmd = A3CTCPConfig::StartGrad;
//	uint64_t* cmdPtr = (uint64_t*)(gradSndBuf.data());
//	cmdPtr[0] = cmd;
	void* bufPtr = static_cast<void*>(gradSndBuf.data());

	LOG4CXX_INFO(logger, "Send start grad sending");
	conn->send(bufPtr, req->expSize);
}

template<typename NetType>
void  A3CTCPClientHandle<NetType>::sendingGrad(void* dataPtr, std::size_t len) {
	GradUpdateReq* req = (GradUpdateReq*)(dataPtr);
	uint64_t index = req->index;
//	uint64_t* cmdPtr = static_cast<uint64_t*>(dataPtr);
//	uint64_t index = cmdPtr[1];

	if (index == gradStr.length()) {
		LOG4CXX_INFO(logger, "Complete sending grad");

//		boost::array<uint64_t, 1> sndBuf;
//		uint64_t* cmdPtr = (uint64_t*)(gradSndBuf.data());
//		cmdPtr[0] = A3CTCPConfig::EndGrad;
		GradUpdateComplete* rsp = A3CTCPCmdFactory::CreateGradComplete(gradSndBuf.data());
		conn->send((void*)(gradSndBuf.data()), rsp->expSize);
	} else {
		if (index != sendGradIndex) {
			LOG4CXX_ERROR(logger, "Data loss in transfer: " << index << " != " << sendGradIndex);

//			boost::array<uint64_t, 1> sndBuf;
//			uint64_t* cmdPtr = (uint64_t*)(gradSndBuf.data());
//			cmdPtr[0] = A3CTCPConfig::GradInterrupt;
			GradUpdateError* rsp = A3CTCPCmdFactory::CreateGradUpdateError(gradSndBuf.data());
			conn->send((void*)(gradSndBuf.data()), rsp->expSize);
		} else {
			LOG4CXX_INFO(logger, "Continue sending grad from " << index);

			uint64_t gradLen = std::min(gradStr.length() - index, (std::size_t)A3CTCPConfig::BufCap - sizeof(GradUpdateRspHd));
//			boost::array<char, A3CTCPConfig::BufCap> sndBuf;
//			uint64_t* sndCmdPtr = (uint64_t*)(gradSndBuf.data());
//			sndCmdPtr[0] = A3CTCPConfig::GradSending;
//			sndCmdPtr[1] = index;
//			sndCmdPtr[2] = gradLen;
			GradUpdateRspHd* rspHd = A3CTCPCmdFactory::CreateGradUpdateRspHd(gradSndBuf.data(), index, gradLen);


//			char* gradPtr = (char*)(sndCmdPtr + 3);
			char* gradPtr = gradSndBuf.data() + sizeof(GradUpdateRspHd);
			memcpy(gradPtr, gradStr.data() + index, gradLen);

			conn->send((void*)(gradSndBuf.data()), rspHd->hd.expSize);

			sendGradIndex += gradLen;
		}
	}
}

template<typename NetType>
void A3CTCPClientHandle<NetType>::completeGrad() {
	for (auto& g: grads) {
		g.fill_(0);
	}
	LOG4CXX_INFO(logger, "End of grad sending");
}

//TODO: sync target at very beginning
template<typename NetType>
void A3CTCPClientHandle<NetType>::syncTarget() {
	std::unique_lock<std::mutex> lock(targetMutex);
	if (targetUpdating) {
		LOG4CXX_ERROR(logger, "target updating in process");
		return;
	}
	targetUpdating = true;

	targetStream.clear();
	targetStream.str("");
	getTargetIndex = 0;

//	uint64_t* cmdPtr = (uint64_t*)(targetSndBuf.data());
//	cmdPtr[0] = A3CTCPConfig::StartTarget;
	TargetSyncStartReq* req = A3CTCPCmdFactory::CreateTargetSyncStartReq(targetSndBuf.data());
	conn->send(static_cast<void*>(targetSndBuf.data()), req->expSize);
	LOG4CXX_INFO(logger, "Start to sync target");

	while (targetUpdating) {
		targetCond.wait(lock);
	}
	LOG4CXX_INFO(logger, "-------------------------------------> End of syncTarget");
}

template<typename NetType>
void A3CTCPClientHandle<NetType>::startUpdateTarget() {
	LOG4CXX_INFO(logger, "Server agreed to update target updating");

//	uint64_t* cmdPtr = (uint64_t*)(targetSndBuf.data());
//	cmdPtr[0] = A3CTCPConfig::TargetSending;
//	cmdPtr[1] = getTargetIndex;
	TargetSyncReq* req = A3CTCPCmdFactory::CreateTargetSyncReq(targetSndBuf.data(), getTargetIndex);
	conn->send((void*)(targetSndBuf.data()), req->hd.expSize);
//	LOG4CXX_INFO(logger, "Sent target request for index " << getTargetIndex);
}

//TODO: conn->send failure
template<typename NetType>
void A3CTCPClientHandle<NetType>::updateTarget(void* data, std::size_t len) {
//	uint64_t* cmdPtr = static_cast<uint64_t*>(data);
//	uint64_t index = cmdPtr[1];
//	uint64_t dataLen = cmdPtr[2];
	TargetSyncRspHd* req = (TargetSyncRspHd*)(data);
	uint64_t index = req->index;
	uint64_t dataLen = req->sndLen;

	if (getTargetIndex != index) {
		LOG4CXX_ERROR(logger, "Transferred size not match " << getTargetIndex << " << != " << index);
//		boost::array<uint64_t, 1> sndBuf;
//		uint64_t* sndCmdPtr = (uint64_t*)(targetSndBuf.data());
//		sndCmdPtr[0] = A3CTCPConfig::TargetInterrupt;
		TargetSyncError* rsp = A3CTCPCmdFactory::CreateTargetSyncError(targetSndBuf.data());
		conn->send(static_cast<void*>(targetSndBuf.data()), rsp->expSize);
	} else {
		LOG4CXX_INFO(logger, "Update target with index = " << index << " dataLen = " << dataLen);

		std::string targetStr((char*)(data) + sizeof(TargetSyncRspHd), dataLen);
		targetStream << targetStr;
		getTargetIndex += dataLen;

//		boost::array<uint64_t, 2> sndBuf;
//		uint64_t* sndCmdPtr = (uint64_t*)(targetSndBuf.data());
//		sndCmdPtr[0] = A3CTCPConfig::TargetSending;
//		sndCmdPtr[1] = getTargetIndex;
		TargetSyncReq* rsp = A3CTCPCmdFactory::CreateTargetSyncReq(targetSndBuf.data(), getTargetIndex);
//		LOG4CXX_INFO(logger, "Client send update target " << getTargetIndex);
		void* bufPtr = static_cast<void*>(targetSndBuf.data());
		conn->send(bufPtr, rsp->hd.expSize);
	}
}

template<typename NetType>
void A3CTCPClientHandle<NetType>::completeTarget() {
	LOG4CXX_INFO(logger, "Target transfer completed expected = " << getTargetIndex << " actual = " << targetStream.str().length());

	std::vector<torch::Tensor> ts;

	try {
		torch::load(ts, targetStream);
		LOG4CXX_INFO(logger, "converted to ts: " << ts.size());

		if(ts.size() != targetParams.size()) {
			LOG4CXX_ERROR(logger, "size of parameters do not match: " << ts.size() << " != " << targetParams.size());
			throw std::invalid_argument("vector size does not match"); //TODO: create an exception class
		}

		for (int i = 0; i < ts.size(); i ++) {
//			if (ts[i].sizes().equals(targetParams[i].sizes())) {
			if (! DummyFuncs::TorchSizesEq(ts[i].sizes(), targetParams[i].sizes())) {
				LOG4CXX_ERROR(logger, "size of parameter do not match " << ts[i].sizes() << " != " << targetParams[i].sizes());
				throw std::invalid_argument("size of parameter does not match");
			}
		}

		torch::NoGradGuard guard;
		std::vector<torch::Tensor> params = net.parameters();
		for (int i = 0; i < params.size(); i ++) {
			params[i].copy_(ts[i]);
		}
		LOG4CXX_INFO(logger, "Target parameters synchronized");
	}catch(std::exception& e) {
		LOG4CXX_ERROR(logger, "Failed to convert tensors: " << e.what());
	}

	std::unique_lock<std::mutex> lock(targetMutex);
	targetUpdating = false;
	targetCond.notify_all();

//	boost::array<uint64, 1> sndBuf;
//	sndBuf[1] = A3CTCPConfig::EndTarget;
//	void bufPtr = static_cast<void*>(sndBuf.data());
//	conn->send(bufPtr, sizeof(uint64_t) * sndBuf.size());
}



template<typename NetType>
void A3CTCPClientHandle<NetType>::sendTest() {
//	uint64_t* cmdPtr = (uint64_t*)(targetSndBuf.data());
//	cmdPtr[0] = A3CTCPConfig::Test;
	TestMsg* msg = A3CTCPCmdFactory::CreateTest(targetSndBuf.data());
	conn->send(static_cast<void*>(targetSndBuf.data()), msg->expSize);
	//TODO: send failure
//	LOG4CXX_INFO(logger, "send test to server");
}

template<typename NetType>
void A3CTCPClientHandle<NetType>::rcvTest(void* data, size_t len) {
//	uint64_t* cmdPtr = (uint64_t*)data;
	TestMsg* msg = (TestMsg*)(data);
	LOG4CXX_INFO(logger, "Received server test response: " << msg->cmd);
}


template<typename NetType>
void A3CTCPClientHandle<NetType>::processRcv(void* bufPtr, size_t totalLen) {
	LOG4CXX_INFO(logger, "Received buffer " << totalLen);
	int bufIndex = 0;

	while (bufIndex < totalLen) {
//	uint64_t* cmdPtr = static_cast<uint64_t*>(dataPtr);
//	uint64_t cmd = cmdPtr[0];
		void* dataPtr = (void*)((char*)bufPtr + bufIndex);
		A3CTCPCmdHd* hd = (A3CTCPCmdHd*)(dataPtr);
		uint64_t cmd = hd->cmd;
		uint64_t len = hd->expSize;
		uint64_t actRemain = totalLen - bufIndex;
		LOG4CXX_INFO(logger, "---------------------> wants " << len << " remains " << actRemain);


		bufIndex += len;

	LOG4CXX_INFO(logger, "Received cmd " << cmd << " len = " << len);

	switch(cmd) {
	case A3CTCPConfig::GradSending:
		sendingGrad(dataPtr, len);
		break;
	case A3CTCPConfig::EndGrad:
		completeGrad();
		break;
	case A3CTCPConfig::GradInterrupt:
		LOG4CXX_ERROR(logger, "TODO grad interrupt");
		break;
	case A3CTCPConfig::StartGrad:
		LOG4CXX_ERROR(logger, "unexpected: StartGrad");
		break;
	case A3CTCPConfig::StartTarget:
		startUpdateTarget();
		break;
	case A3CTCPConfig::TargetSending:
		updateTarget(dataPtr, len);
		break;
	case A3CTCPConfig::EndTarget:
		completeTarget();
		break;
	case A3CTCPConfig::TargetInterrupt:
		LOG4CXX_ERROR(logger, "TODO unexpected: TargetInterrupted");
		break;
	case A3CTCPConfig::Test:
		rcvTest(dataPtr, len);
		break;
	default:
		LOG4CXX_ERROR(logger, "Unexpected cmd: " << cmd);
	}

	}
}
#endif /* INC_A3C_A3CTCPCLIENTHANLE_HPP_ */
