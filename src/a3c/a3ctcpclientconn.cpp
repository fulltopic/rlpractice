/*
 * a3ctcpclientconn.cpp
 *
 *  Created on: Nov 10, 2021
 *      Author: zf
 */

#include "a3c/a3ctcpclientconn.h"
#include "a3c/a3ctcpconfig.h"
#include "a3c/dummyfuncs.h"
#include "a3c/a3ctcpmsghd.h"

std::shared_ptr<A3CTCPClientConn> A3CTCPClientConn::Create(boost::asio::io_service& iio) {
	return std::shared_ptr<A3CTCPClientConn>(new A3CTCPClientConn(iio));
}

A3CTCPClientConn::A3CTCPClientConn(boost::asio::io_service& iio):
		resolver(iio),
		sock(iio),
		serverP(boost::asio::ip::address::from_string(A3CTCPConfig::ServerIp), A3CTCPConfig::ServerPort)
{
//	rcvFunc = std::bind(&DummyFuncs::dummyRcv, dummy,
//			std::placeholders::_1,
//			std::placeholders::_2);

	rcvFunc = DummyFuncs::DummyRcv;
}

A3CTCPClientConn::~A3CTCPClientConn() {
	sock.close();
}

void A3CTCPClientConn::setRcvFunc(std::function<void(void*, std::size_t)> func) {
	rcvFunc = func;
}

void A3CTCPClientConn::start() {
	sock.open(boost::asio::ip::tcp::v4());
	sock.connect(serverP);
	peekRcv();
//	rcv();
}

void A3CTCPClientConn::peekRcv() {
	sock.async_receive( boost::asio::buffer(rcvBuf.data(), sizeof(A3CTCPCmdHd)), boost::asio::ip::tcp::socket::message_peek,
			boost::bind(&A3CTCPClientConn::handlePeekRcv, this->shared_from_this(),
					boost::asio::placeholders::error,
					boost::asio::placeholders::bytes_transferred));
}

void A3CTCPClientConn::handlePeekRcv(const boost::system::error_code& error, std::size_t len) {
	LOG4CXX_DEBUG(logger, "peek cmd hd " << len);

	A3CTCPCmdHd* hd = (A3CTCPCmdHd*)(rcvBuf.data());
	uint64_t expSize = hd->expSize;

	if (expSize <= A3CTCPConfig::BufCap) {
		LOG4CXX_DEBUG(logger, "expect to receive msg with size " << expSize);
		sock.async_receive( boost::asio::buffer(rcvBuf.data(), expSize),
				boost::bind(&A3CTCPClientConn::handleRcv, this->shared_from_this(),
						boost::asio::placeholders::error,
						boost::asio::placeholders::bytes_transferred));
	} else {
		LOG4CXX_ERROR(logger, "Error in command parse " << expSize);
	}
}
//
//void A3CTCPClientConn::rcv() {
//	sock.async_receive( boost::asio::buffer(rcvBuf.data(), A3CTCPConfig::BufCap),
//			boost::bind(&A3CTCPClientConn::handleRcv, this->shared_from_this(),
//					boost::asio::placeholders::error,
//					boost::asio::placeholders::bytes_transferred));
//}

void A3CTCPClientConn::handleRcv(const boost::system::error_code& error, std::size_t len) {
	if (!error) {
		//TODO: handle process
		rcvFunc(static_cast<void*>(rcvBuf.data()), len);
	} else {
		LOG4CXX_ERROR(logger, "Failure in receiving: " << error);
	}

//	rcv();
	peekRcv();
//	LOG4CXX_INFO(logger, "Received " << len);
//
//	uint64_t* cmdPtr = (uint64_t*)rcvBuf.data();
//	if ((*cmdPtr) == 10) {
//		uint64_t index = *(cmdPtr + 1);
//		LOG4CXX_INFO(logger, "To send from index " << index);
//		auto bufLen = std::min(tmpFileStr.length() - index, (std::size_t)A3CTCPConfig::BufCap - sizeof(uint64_t) * 3);
//
////		std::stringstream ss;
//		boost::array<char, A3CTCPConfig::BufCap> sndBuf;
//		if (bufLen == 0) {
//			uint64_t cmd = 11;
//			uint64_t* sndData = (uint64_t*)((void*)(sndBuf.data()));
//			sndData[0] = cmd;
//
//			sock.send(boost::asio::buffer(sndBuf.data(), sizeof(uint64_t)));
////			ss << cmd;
//		} else {
//			uint64_t cmd = 10;
////			ss << cmd;
////			ss << index;
//			uint64_t* sndData = (uint64_t*)((void*)(sndBuf.data()));
//			sndData[0] = cmd;
//			sndData[1] = index;
//			sndData[2] = bufLen;
//			void* dataPtr = (void*)(sndData + 3);
//			memcpy(dataPtr, (void*)(tmpFileStr.data() + index), bufLen);
//
//			sock.send(boost::asio::buffer(sndBuf.data(), bufLen + sizeof(uint64_t) * 3));
////			std::string str(tmpFileStr.data() + index, sendLen);
////			ss << str;
//		}
//
////		std::string sendStr = ss.str();
////		sock.send(boost::asio::buffer(sendStr.data(), sendStr.length()));
//
//		rcv();
//	} else {
//		LOG4CXX_INFO(logger, "Received unexpected :" << *cmdPtr);
//	}
}

void A3CTCPClientConn::handleSnd(const boost::system::error_code& error, std::size_t len) {
	if (!error) {
//		LOG4CXX_INFO(logger, "Sent something " << len);
	} else {
		LOG4CXX_ERROR(logger, "Failure in receiving: " << error);
	}

}

bool A3CTCPClientConn::send(void* data, std::size_t len) {
//	auto sndLen = sock.send(boost::asio::buffer(data, len));
//	if (sndLen == len) {
//		return true;
//	} else {
//		LOG4CXX_ERROR(logger, "Not sent all data");
//		return false;
//	}

    boost::asio::async_write(sock, boost::asio::buffer(data, len),
        boost::bind(&A3CTCPClientConn::handleSnd, this->shared_from_this(),
          boost::asio::placeholders::error,
          boost::asio::placeholders::bytes_transferred
		  ));

    return true;
}

bool A3CTCPClientConn::send(std::vector<torch::Tensor> ts) {
//	std::ostringstream os;
//	torch::save(ts, os);
//	std::string strBuf = os.str();
//
//	std::size_t index = 0;
//
//	while (index < strBuf.length()) {
//		try {
//			auto bufLen = std::min(strBuf.length() - index, (std::size_t)A3CTCPConfig::BufCap);
//			LOG4CXX_INFO(logger, "bufLen = " << bufLen << " index = " << index);
//			auto sentBytes = sock.send(boost::asio::buffer(strBuf.data() + index, bufLen));
//			LOG4CXX_INFO(logger, "Sent: " << sentBytes << " to " << serverP );
//			index += sentBytes;
////			return true;
//		} catch (boost::system::system_error& e) {
//			LOG4CXX_ERROR(logger, "Sent error " << e.what());
//			return false;
//		}
//	}
//	LOG4CXX_INFO(logger, "Sent: " << index);
//	return true;

	tmpFileSs.clear();
	torch::save(ts, tmpFileSs);
	tmpFileStr = tmpFileSs.str();

//	std::stringstream ss;
//	uint64_t cmd = 9;
//	ss << cmd;
//	std::string cmdStr = ss.str();
	uint64_t cmd = 9;

	sock.send(boost::asio::buffer((void*)(&cmd), sizeof(uint64_t)));
	return true;
}
