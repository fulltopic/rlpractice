/*
 * a3ctcpserverqhdl.hpp
 *
 *  Created on: Dec 2, 2021
 *      Author: zf
 */

#ifndef INC_A3C_A3CTCPSERVERQHDL_HPP_
#define INC_A3C_A3CTCPSERVERQHDL_HPP_



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
#include "a3cgradque.h"

template<typename NetType>
class A3CTCPServerQHandle : public std::enable_shared_from_this<A3CTCPServerQHandle<NetType>>, A3CTCPServerHandleInterface {
private:
	A3CTCPServerQHandle(boost::asio::io_service& iio, NetType& iNet, A3CGradQueue& iQ);

	NetType& net;
//	OptType& opt;
	A3CGradQueue& q;
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
	virtual ~A3CTCPServerQHandle();
	A3CTCPServerQHandle(const A3CTCPServerQHandle&) = delete;

	static std::shared_ptr<A3CTCPServerQHandle<NetType>> Create(boost::asio::io_service& iio, NetType& iNet, A3CGradQueue& iQ);
	static std::shared_ptr<A3CTCPServerHandleInterface> CreateInterface(boost::asio::io_service& iio, NetType& iNet, A3CGradQueue& iQ);
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

template<typename NetType>
A3CTCPServerQHandle<NetType>::A3CTCPServerQHandle(boost::asio::io_service& iio, NetType& iNet, A3CGradQueue& iQ)
	: net(iNet),
	  q(iQ)
{
	conn = A3CTCPServerConn::Create(iio);
}

template<typename NetType>
A3CTCPServerQHandle<NetType>::~A3CTCPServerQHandle() {
}

template<typename NetType>
boost::asio::ip::tcp::socket&  A3CTCPServerQHandle<NetType>::getSock() {
	return conn->getSock();
}

template<typename NetType>
uint64_t A3CTCPServerQHandle<NetType>::getUpdateNum() {
	return updateNum;
}

template<typename NetType>
std::shared_ptr<A3CTCPServerQHandle<NetType>> A3CTCPServerQHandle<NetType>::Create(boost::asio::io_service& iio, NetType& iNet, A3CGradQueue& iQ) {
	return std::shared_ptr<A3CTCPServerQHandle<NetType>>(new A3CTCPServerQHandle<NetType>(iio, iNet, iQ));
}

template<typename NetType>
std::shared_ptr<A3CTCPServerHandleInterface> A3CTCPServerQHandle<NetType>::CreateInterface(boost::asio::io_service& iio, NetType& iNet, A3CGradQueue& iQ) {
	A3CTCPServerHandleInterface* obj = new A3CTCPServerQHandle<NetType>(iio, iNet, iQ);
	return std::shared_ptr<A3CTCPServerHandleInterface>(obj);
}

template<typename NetType>
void A3CTCPServerQHandle<NetType>::start() {
	 std::function<void(void*, std::size_t len)> func = std::bind(&A3CTCPServerQHandle<NetType>::processRcv, this,
			 std::placeholders::_1,
			 std::placeholders::_2);

	 conn->setRcvFunc(func);
	 conn->start();
}

template<typename NetType>
void A3CTCPServerQHandle<NetType>::syncGrad() {
	LOG4CXX_DEBUG(logger, "Server agreed to sync grad");

	gradStream.clear();
	gradStream.str("");
	getGradIndex = 0;

	GradUpdateReq* req = A3CTCPCmdFactory::CreateGradUpdateReq(gradSndBuf.data(), getGradIndex);

	conn->send(static_cast<void*>(gradSndBuf.data()), req->hd.expSize);
	//TODO: send failure
}

template<typename NetType>
void A3CTCPServerQHandle<NetType>::sendingGrad(void* dataPtr, std::size_t len) {
	GradUpdateRspHd* rspHd = (GradUpdateRspHd*)(dataPtr);
	uint64_t index = rspHd->index;
	uint64_t dataLen = rspHd->dataLen;

	if (index != getGradIndex) {
		LOG4CXX_ERROR(logger, "data lost in transfer: " << index << " != " << getGradIndex);

		GradUpdateError* req = A3CTCPCmdFactory::CreateGradUpdateError(gradSndBuf.data());
		conn->send((void*)(gradSndBuf.data()), sizeof(GradUpdateError));
	} else {
		LOG4CXX_DEBUG(logger, "Server receiving grad: " << len);

		char* bufPtr = (char*)dataPtr + sizeof(GradUpdateRspHd);
		std::string gradStr(bufPtr, dataLen);
		gradStream << gradStr;

		getGradIndex += dataLen;

		GradUpdateReq* req = A3CTCPCmdFactory::CreateGradUpdateReq(gradSndBuf.data(), getGradIndex);
		conn->send((void*)(gradSndBuf.data()), req->hd.expSize);
	}
}

template<typename NetType>
void A3CTCPServerQHandle<NetType>::completeGrad() {
	updateNum ++;

	std::vector<torch::Tensor> ts;

	try {
		torch::load(ts, gradStream);
		LOG4CXX_INFO(logger, "Completed grad transfer, converted to ts: " << ts.size());

		const std::vector<torch::Tensor> params = net.parameters();

		if(ts.size() != params.size()) {
			LOG4CXX_ERROR(logger, "size of parameters do not match: " << ts.size() << " != " << params.size());
			throw std::invalid_argument("vector size does not match");
		}

		for (int i = 0; i < ts.size(); i ++) {
			if (ts[i].numel() == 0) {
				continue;
			}
			if (!DummyFuncs::TorchSizesEq(ts[i].sizes(), params[i].sizes())) {
				LOG4CXX_ERROR(logger, "size of parameter do not match " << ts[i].sizes() << " != " << params[i].sizes());
				throw std::invalid_argument("size of parameter does not match");
			}
		}

		q.push(ts);
	}catch(std::exception& e) {
		LOG4CXX_ERROR(logger, "Failed to convert tensors: " << e.what());
	}

	GradUpdateComplete* sndReq = A3CTCPCmdFactory::CreateGradComplete(gradSndBuf.data());
	conn->send((void*)(gradSndBuf.data()), sizeof(GradUpdateComplete));
//	LOG4CXX_INFO(logger, "Sent end of grad update");
}

template<typename NetType>
void A3CTCPServerQHandle<NetType>::startUpdateTarget() {

	targetStream.clear();
	targetStream.str("");
//	LOG4CXX_INFO(logger, "write string ");
//	LOG4CXX_INFO(logger, "write: " << targetStream.str().length());
	std::vector<torch::Tensor> params = net.parameters();
	torch::save(net.parameters(), targetStream);
	targetStr = targetStream.str();
	sendTargetIndex = 0;
	LOG4CXX_DEBUG(logger, "Start to update target, written target stream length " << targetStr.length());

	TargetSyncStartRsp* rsp = A3CTCPCmdFactory::CreateTargetSyncStartRsp(targetSndBuf.data());
	conn->send((void*)(targetSndBuf.data()), rsp->expSize);
//	LOG4CXX_INFO(logger, "Start to update target");
}

template<typename NetType>
void A3CTCPServerQHandle<NetType>::updateTarget(void* dataPtr, std::size_t len) {
	TargetSyncReq* req = (TargetSyncReq*)dataPtr;
	uint64_t index = req->index;

	if (index != sendTargetIndex) {
		LOG4CXX_ERROR(logger, "Data lost in trasfer: " << index << " != " << sendTargetIndex);

		TargetSyncError* rsp = A3CTCPCmdFactory::CreateTargetSyncError(targetSndBuf.data());
		//TODO: Debug, extend size
		conn->send(static_cast<void*>(targetSndBuf.data()), rsp->expSize);
	} else {
		if (targetStr.length() == index) {
			TargetSyncComplete* rsp = A3CTCPCmdFactory::CreateTargetSyncComplete(targetSndBuf.data());
			conn->send((void*)(targetSndBuf.data()), rsp->expSize);

				LOG4CXX_INFO(logger, "End of target sync");
		} else {
			uint64_t sndLen = std::min(targetStr.length() - index, A3CTCPConfig::BufCap - sizeof(TargetSyncRspHd));
			LOG4CXX_DEBUG(logger, "update target from " << index << " with " << sndLen);
			TargetSyncRspHd* rspHd = A3CTCPCmdFactory::CreateTargetSyncRspHd(targetSndBuf.data(), index, sndLen);
			memcpy(targetSndBuf.data() + sizeof(TargetSyncRspHd), targetStr.data() + index, sndLen);
			conn->send((void*)(targetSndBuf.data()), rspHd->hd.expSize);
//			LOG4CXX_INFO(logger, "Sent target: " << index << ", " << sndLen);
			sendTargetIndex += sndLen;
		}
	}
}

template<typename NetType>
void A3CTCPServerQHandle<NetType>::rcvTest(void* data, std::size_t len) {
	TestMsg* req = (TestMsg*)(data);
	LOG4CXX_INFO(logger, "received test from client: " << req->cmd);

	TestMsg* rsp = A3CTCPCmdFactory::CreateTest(targetSndBuf.data());
	conn->send((void*)(targetSndBuf.data()), rsp->expSize);
	LOG4CXX_INFO(logger, "sent back test response: " << rsp->cmd);
}

template<typename NetType>
void A3CTCPServerQHandle<NetType>::processRcv(void* bufPtr, std::size_t bufLen) {
	LOG4CXX_DEBUG(logger, "server received buf " << bufLen);
	std::size_t bufIndex = 0;

	while (bufIndex < bufLen) {
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


#endif /* INC_A3C_A3CTCPSERVERQHDL_HPP_ */
