/*
 * a3ctcpserverhandle.hpp
 *
 *  Created on: Nov 12, 2021
 *      Author: zf
 */

#ifndef INC_A3C_A3CTCPSERVERHANDLE_HPP_
#define INC_A3C_A3CTCPSERVERHANDLE_HPP_


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

#include <torch/torch.h>

#include "a3c/a3ctcpconfig.h"
#include "a3c/a3ctcpserverconn.h"
#include "a3ctcpmsghds.h"
#include "a3ctcpserverhandleinterface.h"

template<typename NetType, typename OptType>
class A3CTCPServerHandle : public std::enable_shared_from_this<A3CTCPServerHandle<NetType, OptType>>, A3CTCPServerHandleInterface {
private:
	A3CTCPServerHandle(boost::asio::io_service& iio, NetType& iNet, OptType& iOpt);

	NetType& net;
	OptType& opt;
	std::shared_ptr<A3CTCPServerConn> conn;
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("a3cserverhandle");

//	std::vector<torch::Tensor> grads;
	std::vector<torch::Tensor> targetParams;

	std::stringstream gradStream;
	std::string gradStr;
	uint64_t getGradIndex = 0;

	boost::array<char, A3CTCPConfig::BufCap> targetSndBuf; //TODO: buffer size to be extended
	boost::array<char, A3CTCPConfig::BufCap> gradSndBuf; //TODO: buffer size to be extended

	std::stringstream targetStream;
	std::stringstream tmpStream;
	std::string targetStr;
	uint64_t sendTargetIndex = 0;

	uint64_t updateNum= 0;
public:
	virtual ~A3CTCPServerHandle();
	A3CTCPServerHandle(const A3CTCPServerHandle&) = delete;

	static std::shared_ptr<A3CTCPServerHandle<NetType, OptType>> Create(boost::asio::io_service& iio, NetType& iNet, OptType& iOpt);
	static std::shared_ptr<A3CTCPServerHandleInterface> CreateInterface(boost::asio::io_service& iio, NetType& iNet, OptType& iOpt);
	virtual void start();
	virtual boost::asio::ip::tcp::socket& getSock();
	void syncGrad();
//	void getGrad();
	void sendingGrad(void* dataPtr, size_t len);
	void completeGrad();

	void startUpdateTarget();
//	void updateTarget(void* data, std::size_t len);
//	void completeTarget();
	void updateTarget(void* dataPtr, size_t len);

	void rcvTest(void* dataPtr, size_t len);

	void processRcv(void* dataPtr, size_t len);

	uint64_t getUpdateNum();
};

template<typename NetType, typename OptType>
A3CTCPServerHandle<NetType, OptType>::A3CTCPServerHandle(boost::asio::io_service& iio, NetType& iNet, OptType& iOpt)
	: net(iNet),
	  opt(iOpt)
{
	conn = A3CTCPServerConn::Create(iio);
}

template<typename NetType, typename OptType>
A3CTCPServerHandle<NetType, OptType>::~A3CTCPServerHandle() {
}

template<typename NetType, typename OptType>
boost::asio::ip::tcp::socket&  A3CTCPServerHandle<NetType, OptType>::getSock() {
	return conn->getSock();
}

template<typename NetType, typename OptType>
uint64_t A3CTCPServerHandle<NetType, OptType>::getUpdateNum() {
	return updateNum;
}

template<typename NetType, typename OptType>
std::shared_ptr<A3CTCPServerHandle<NetType, OptType>> A3CTCPServerHandle<NetType, OptType>::Create(boost::asio::io_service& iio, NetType& iNet, OptType& iOpt) {
	return std::shared_ptr<A3CTCPServerHandle<NetType, OptType>>(new A3CTCPServerHandle<NetType, OptType>(iio, iNet, iOpt));
}

template<typename NetType, typename OptType>
std::shared_ptr<A3CTCPServerHandleInterface> A3CTCPServerHandle<NetType, OptType>::CreateInterface(boost::asio::io_service& iio, NetType& iNet, OptType& iOpt) {
	A3CTCPServerHandleInterface* obj = new A3CTCPServerHandle<NetType, OptType>(iio, iNet, iOpt);
	return std::shared_ptr<A3CTCPServerHandleInterface>(obj);
//	return std::shared_ptr<A3CTCPServerHandleInterface>(new A3CTCPServerHandle<NetType, OptType>(iio, iNet, iOpt));
}

template<typename NetType, typename OptType>
void A3CTCPServerHandle<NetType, OptType>::start() {
	 std::function<void(void*, std::size_t len)> func = std::bind(&A3CTCPServerHandle<NetType, OptType>::processRcv, this,
			 std::placeholders::_1,
			 std::placeholders::_2);

	 conn->setRcvFunc(func);
	 conn->start();
}

template<typename NetType, typename OptType>
void A3CTCPServerHandle<NetType, OptType>::syncGrad() {
	LOG4CXX_DEBUG(logger, "Server agreed to sync grad");

	gradStream.clear();
	gradStream.str("");
	getGradIndex = 0;

//	boost::array<uint64_t, 2> cmdBuf;
//	uint64_t* cmdPtr = (uint64_t*)(gradSndBuf.data());
//	cmdPtr[0] = A3CTCPConfig::GradSending;
//	cmdPtr[1] = getGradIndex;
	GradUpdateReq* req = A3CTCPCmdFactory::CreateGradUpdateReq(gradSndBuf.data(), getGradIndex);

	conn->send(static_cast<void*>(gradSndBuf.data()), req->hd.expSize);
	//TODO: send failure
}

template<typename NetType, typename OptType>
void A3CTCPServerHandle<NetType, OptType>::sendingGrad(void* dataPtr, std::size_t len) {
//	uint64_t* cmdPtr = static_cast<uint64_t*>(dataPtr);
//	uint64_t cmd = cmdPtr[0];
//	uint64_t index = cmdPtr[1];
//	uint64_t dataLen = cmdPtr[2];
	GradUpdateRspHd* rspHd = (GradUpdateRspHd*)(dataPtr);
	uint64_t index = rspHd->index;
	uint64_t dataLen = rspHd->dataLen;

	if (index != getGradIndex) {
		LOG4CXX_ERROR(logger, "data lost in transfer: " << index << " != " << getGradIndex);

//		uint64_t* sndCmdPtr = (uint64_t*)(gradSndBuf.data());
//		sndCmdPtr[0] = A3CTCPConfig::GradInterrupt;
		GradUpdateError* req = A3CTCPCmdFactory::CreateGradUpdateError(gradSndBuf.data());
		conn->send((void*)(gradSndBuf.data()), sizeof(GradUpdateError));
	} else {
		LOG4CXX_DEBUG(logger, "Server receiving grad: " << len);

//		char* bufPtr = (char*)(cmdPtr + 3);
		char* bufPtr = (char*)dataPtr + sizeof(GradUpdateRspHd);
		std::string gradStr(bufPtr, dataLen);
		gradStream << gradStr;

		getGradIndex += dataLen;

//		boost::array<uint64_t, 2> sndBuf;
//		uint64_t* sndBuf = (uint64_t*)(gradSndBuf.data());
//		sndBuf[0] = A3CTCPConfig::GradSending;
//		sndBuf[1] = getGradIndex;
		GradUpdateReq* req = A3CTCPCmdFactory::CreateGradUpdateReq(gradSndBuf.data(), getGradIndex);
		conn->send((void*)(gradSndBuf.data()), req->hd.expSize);
	}
}

template<typename NetType, typename OptType>
void A3CTCPServerHandle<NetType, OptType>::completeGrad() {
	updateNum ++;

	opt.zero_grad();
	std::vector<torch::Tensor> ts;

	try {
		torch::load(ts, gradStream);
		LOG4CXX_INFO(logger, "Completed grad transfer, converted to ts: " << ts.size());

		std::vector<torch::Tensor> params = net.parameters();

		if(ts.size() != params.size()) {
			LOG4CXX_ERROR(logger, "size of parameters do not match: " << ts.size() << " != " << params.size());
			throw std::invalid_argument("vector size does not match");
		}

		for (int i = 0; i < ts.size(); i ++) {
			if (ts[i].numel() == 0) {
				continue;
			}
//			if (ts[i].sizes().equals(params[i].sizes())) {
			if (!DummyFuncs::TorchSizesEq(ts[i].sizes(), params[i].sizes())) {
				LOG4CXX_ERROR(logger, "size of parameter do not match " << ts[i].sizes() << " != " << params[i].sizes());
				throw std::invalid_argument("size of parameter does not match");
			}
		}

		for (int i = 0; i < params.size(); i ++) {
			if (ts[i].numel() == 0) {
				LOG4CXX_DEBUG(logger, "No grad of layer " << i);
				continue;
			}

			params[i].mutable_grad() = ts[i]; //TODO: grad valid?
//			LOG4CXX_INFO(logger, "server get grad \n" << net.parameters()[i].grad());
		}

		torch::nn::utils::clip_grad_norm_(net.parameters(), 0.1);
		opt.step();
		//TODO: no zero_grad?
	}catch(std::exception& e) {
		LOG4CXX_ERROR(logger, "Failed to convert tensors: " << e.what());
	}

//	boost::array<uint64_t, 1> sndBuf;
//	sndBuf[0] = A3CTCPConfig::EndGrad;
//	void* bufPtr = (void*)(sndBuf.data());
//	conn->send(bufPtr, sizeof(uint64_t) * sndBuf.size());
	GradUpdateComplete* sndReq = A3CTCPCmdFactory::CreateGradComplete(gradSndBuf.data());
	conn->send((void*)(gradSndBuf.data()), sizeof(GradUpdateComplete));
//	LOG4CXX_INFO(logger, "Sent end of grad update");
}

const size_t fakeLen = 10;
template<typename NetType, typename OptType>
void A3CTCPServerHandle<NetType, OptType>::startUpdateTarget() {
//	LOG4CXX_INFO(logger, "test save");
//	std::stringstream tmpStream;
//	tmpStream.clear();
//	torch::save(net.parameters(), tmpStream);
//	tmpStream << "Testtest";
//	LOG4CXX_INFO(logger, "test save end " << tmpStream.str());
//	LOG4CXX_INFO(logger, "test save: " << tmpStream.str().length());

	targetStream.clear();
	targetStream.str("");
//	targetStream << "testtest" << std::endl;
//	LOG4CXX_INFO(logger, "write string ");
//	LOG4CXX_INFO(logger, "write: " << targetStream.str().length());
	std::vector<torch::Tensor> params = net.parameters();
//	torch::save(net.parameters(), "./test.pt");
	torch::save(net.parameters(), targetStream);
	targetStr = targetStream.str();
	sendTargetIndex = 0;
	LOG4CXX_DEBUG(logger, "Start to update target, written target stream length " << targetStr.length());

//	uint64_t* cmdPtr = (uint64_t*)(targetSndBuf.data());
//	cmdPtr[0] = A3CTCPConfig::StartTarget;
//	//TODO: Debug, extend size
//	conn->send(static_cast<void*>(targetSndBuf.data()), sizeof(uint64_t) * fakeLen);
	TargetSyncStartRsp* rsp = A3CTCPCmdFactory::CreateTargetSyncStartRsp(targetSndBuf.data());
	conn->send((void*)(targetSndBuf.data()), rsp->expSize);
//	LOG4CXX_INFO(logger, "Start to update target");
}

template<typename NetType, typename OptType>
void A3CTCPServerHandle<NetType, OptType>::updateTarget(void* dataPtr, std::size_t len) {
//	uint64_t* cmdPtr = (uint64_t*)(dataPtr);
//	uint64_t index = cmdPtr[1];
	TargetSyncReq* req = (TargetSyncReq*)dataPtr;
	uint64_t index = req->index;

	if (index != sendTargetIndex) {
		LOG4CXX_ERROR(logger, "Data lost in trasfer: " << index << " != " << sendTargetIndex);

//		uint64_t* cmdPtr = (uint64_t*)(targetSndBuf.data());
//		cmdPtr[0] = A3CTCPConfig::TargetInterrupt;
		TargetSyncError* rsp = A3CTCPCmdFactory::CreateTargetSyncError(targetSndBuf.data());
		//TODO: Debug, extend size
		conn->send(static_cast<void*>(targetSndBuf.data()), rsp->expSize);
	} else {
		if (targetStr.length() == index) {
//				uint64_t* sndCmdPtr = (uint64_t*)(targetSndBuf.data());
//				sndCmdPtr[0] = A3CTCPConfig::EndTarget;
				//TODO: Debug, extend size
//				conn->send((void*)(targetSndBuf.data()), sizeof(uint64_t) * fakeLen);
			TargetSyncComplete* rsp = A3CTCPCmdFactory::CreateTargetSyncComplete(targetSndBuf.data());
			conn->send((void*)(targetSndBuf.data()), rsp->expSize);

				LOG4CXX_INFO(logger, "End of target sync");
		} else {
			uint64_t sndLen = std::min(targetStr.length() - index, A3CTCPConfig::BufCap - sizeof(TargetSyncRspHd));
			LOG4CXX_DEBUG(logger, "update target from " << index << " with " << sndLen);
//			boost::array<char, A3CTCPConfig::BufCap> sndBuf;
//			uint64_t* sndCmdPtr = (uint64_t*)(targetSndBuf.data());
//			sndCmdPtr[0] = A3CTCPConfig::TargetSending;
//			sndCmdPtr[1] = index;
//			sndCmdPtr[2] = sndLen;
//			memcpy((char*)(sndCmdPtr + 3), targetStr.data() + index, sndLen);
			TargetSyncRspHd* rspHd = A3CTCPCmdFactory::CreateTargetSyncRspHd(targetSndBuf.data(), index, sndLen);
			memcpy(targetSndBuf.data() + sizeof(TargetSyncRspHd), targetStr.data() + index, sndLen);
			conn->send((void*)(targetSndBuf.data()), rspHd->hd.expSize);
//			conn->send(static_cast<void*>(targetSndBuf.data()), sndLen + sizeof(uint64_t) * 3);
//			LOG4CXX_INFO(logger, "Sent target: " << index << ", " << sndLen);
			sendTargetIndex += sndLen;
		}
	}
}

template<typename NetType, typename OptType>
void A3CTCPServerHandle<NetType, OptType>::rcvTest(void* data, std::size_t len) {
//	uint64_t* cmdPtr = static_cast<uint64_t*>(data);
	TestMsg* req = (TestMsg*)(data);
	LOG4CXX_INFO(logger, "received test from client: " << req->cmd);

//	uint64_t* sndCmdPtr = (uint64_t*)(targetSndBuf.data());
//	sndCmdPtr[0] = A3CTCPConfig::Test;
//	conn->send((void*)(targetSndBuf.data()), sizeof(uint64_t));
	TestMsg* rsp = A3CTCPCmdFactory::CreateTest(targetSndBuf.data());
	conn->send((void*)(targetSndBuf.data()), rsp->expSize);
	LOG4CXX_INFO(logger, "sent back test response: " << rsp->cmd);
}

template<typename NetType, typename OptType>
void A3CTCPServerHandle<NetType, OptType>::processRcv(void* bufPtr, std::size_t bufLen) {
	LOG4CXX_DEBUG(logger, "server received buf " << bufLen);
	std::size_t bufIndex = 0;

	while (bufIndex < bufLen) {
//	uint64_t* cmdPtr = static_cast<uint64_t*>(data);
//	uint64_t cmd = cmdPtr[0];
		void* data = (void*)((char*)bufPtr + bufIndex);
		A3CTCPCmdHd* req = (A3CTCPCmdHd*)(data);
		auto cmd = req->cmd;
		auto len = req->expSize;

		bufIndex += len;

	LOG4CXX_DEBUG(logger, "server received cmd " << cmd);

	switch(cmd) {
	case A3CTCPConfig::StartGrad:
		syncGrad();
		break;
	case A3CTCPConfig::GradSending:
		sendingGrad(data, len);
		break;
	case A3CTCPConfig::EndGrad:
		completeGrad();
		break;
	case A3CTCPConfig::GradInterrupt:
		LOG4CXX_ERROR(logger, "TODO Grad transfer interrupted");
		break;
	case A3CTCPConfig::StartTarget:
		startUpdateTarget();
		break;
	case A3CTCPConfig::TargetSending:
		updateTarget(data, len);
		break;
	case A3CTCPConfig::EndTarget:
		LOG4CXX_INFO(logger, "client notified EndTarget");
		break;
	case A3CTCPConfig::TargetInterrupt:
		LOG4CXX_ERROR(logger, "TODO Target transfer interrupted");
		break;
	case A3CTCPConfig::Test:
		rcvTest(data, len);
		break;
	default:
		LOG4CXX_ERROR(logger, "unexpected cmd: " << cmd);
	}
	}
}
#endif /* INC_A3C_A3CTCPSERVERHANDLE_HPP_ */
