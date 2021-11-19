/*
 * a3ctcpserverconn.cpp
 *
 *  Created on: Nov 10, 2021
 *      Author: zf
 */


#include "a3c/a3ctcpserver.h"
#include "a3c/a3ctcpserverconn.h"
#include "a3c/a3ctcpconfig.h"
#include "a3c/a3ctcpmsghd.h"

std::shared_ptr<A3CTCPServerConn> A3CTCPServerConn::Create(boost::asio::io_service& iio) {
	return std::shared_ptr<A3CTCPServerConn>(new A3CTCPServerConn(iio));
}

A3CTCPServerConn::A3CTCPServerConn(boost::asio::io_service& iio):
		sock(iio)
{
	rcvFunc = DummyFuncs::DummyRcv;
}

void A3CTCPServerConn::setRcvFunc(std::function<void(void*, std::size_t len)> func) {
	rcvFunc = func;
}

boost::asio::ip::tcp::socket& A3CTCPServerConn::getSock() {
	return sock;
}

void A3CTCPServerConn::start() {
//	rcv();
	peekRcv();
}

//void A3CTCPServerConn::rcv() {
//	sock.async_receive( boost::asio::buffer(rcvBuf),
//			boost::bind(&A3CTCPServerConn::handleRcv, this->shared_from_this(),
//			boost::asio::placeholders::error,
//			boost::asio::placeholders::bytes_transferred));
//}

void A3CTCPServerConn::peekRcv() {
	sock.async_receive( boost::asio::buffer(rcvBuf.data(), sizeof(A3CTCPCmdHd)), boost::asio::ip::tcp::socket::message_peek,
			boost::bind(&A3CTCPServerConn::handlePeekRcv, this->shared_from_this(),
					boost::asio::placeholders::error,
					boost::asio::placeholders::bytes_transferred));
}

void A3CTCPServerConn::handlePeekRcv(const boost::system::error_code& error, std::size_t len) {
	LOG4CXX_INFO(logger, "peek cmd hd " << len);

	A3CTCPCmdHd* hd = (A3CTCPCmdHd*)(rcvBuf.data());
	uint64_t expSize = hd->expSize;

	if (expSize <= A3CTCPConfig::BufCap) {
		LOG4CXX_INFO(logger, "expect to receive msg with size " << expSize);
		sock.async_receive( boost::asio::buffer(rcvBuf.data(), expSize),
				boost::bind(&A3CTCPServerConn::handleRcv, this->shared_from_this(),
						boost::asio::placeholders::error,
						boost::asio::placeholders::bytes_transferred));
	} else {
		LOG4CXX_ERROR(logger, "Error in command parse " << expSize);
	}
}

void A3CTCPServerConn::handleRcv(const boost::system::error_code& error, std::size_t len) {
//	if (!error || error == boost::asio::error::message_size) {
//
//		LOG4CXX_INFO(logger, "Received: " << len << ", " << totalRcv);
//		if (len >= A3CTCPConfig::BufCap) {
//			LOG4CXX_INFO(logger, "Received bulk");
//			std::string data(rcvBuf.data(), len);
//			ss << data;
//			totalRcv += len;
//
//			rcv();
//		} else {
//			std::string data(rcvBuf.data(), len);
//			ss << data;
//			totalRcv += len;
//			LOG4CXX_INFO(logger, "Received last: " << len << ", " << totalRcv);
//			LOG4CXX_INFO(logger, "End of rcv");
//
//			try {
//			std::vector<torch::Tensor> ts;
//			torch::load(ts, ss);
//			LOG4CXX_INFO(logger, "ts size: " << ts.size());
//			for (const auto& t: ts) {
//				LOG4CXX_INFO(logger, "t: " << t.sizes());
//			}
//			}catch (std::exception& e) {
//				LOG4CXX_ERROR(logger, "Failed to cast tensor as " << e.what());
//			}
//			totalRcv = 0;
//		}
//	} else {
//		LOG4CXX_ERROR(logger, "Failed to receive message for: " << error);
//	}

//	if (!error || error == boost::asio::error::message_size) {
//		LOG4CXX_INFO(logger, "Received: " << len << ", " << totalRcv);
//		if (len >= 0) {
//			LOG4CXX_INFO(logger, "Received bulk");
//			std::string data(rcvBuf.data(), len);
//			ss << data;
//			totalRcv += len;
//
//			if (totalRcv < 6742044) {
//				rcv();
//			} else {
//				LOG4CXX_INFO(logger, "End of rcv");
//
//				try {
//				std::vector<torch::Tensor> ts;
//				torch::load(ts, ss);
//				LOG4CXX_INFO(logger, "ts size: " << ts.size());
//				for (const auto& t: ts) {
//					LOG4CXX_INFO(logger, "t: " << t.sizes());
//				}
//				}catch (std::exception& e) {
//					LOG4CXX_ERROR(logger, "Failed to cast tensor as " << e.what());
//				}
//				totalRcv = 0;
//				//TODO reset ss after ts processed
//			}
//		} else {
//			LOG4CXX_ERROR(logger, "No content received");
//		}
//	} else {
//		LOG4CXX_ERROR(logger, "Failed to receive message for: " << error);
//	}
//
//	if (!error || error == boost::asio::error::message_size) {
//		LOG4CXX_INFO(logger, "Received started " << len);
//		if (len < A3CTCPConfig::BufCap) {
//			LOG4CXX_INFO(logger, "Simple message ");
//			std::string data(rcvBuf.data(), len);
//			ss << data;
//			try {
//				std::vector<torch::Tensor> ts;
//				torch::load(ts, ss);
//				LOG4CXX_INFO(logger, "ts size " << ts.size());
//				for (const auto& t: ts) {
//					LOG4CXX_INFO(logger, "t: " << t.sizes());
//				}
//			}catch(std::exception& e) {
//				LOG4CXX_ERROR(logger, "Failed to cast tensor as " << e.what());
//			}
//		} else {
//			LOG4CXX_INFO(logger, "large bulk, tbc");
//			std::string data(rcvBuf.data(), len);
////			bs << data;
//			totalRcv += len;
//
//			boost::asio::async_read(sock, bs,
//					boost::asio::transfer_at_least(1),
//					boost::bind(&A3CTCPServerConn::handleRead, this->shared_from_this(),
//							boost::asio::placeholders::error));
//		}
//	} else {
//		LOG4CXX_ERROR(logger, "Failed to receive message for: " << error);
//	}

//	if (!error || error == boost::asio::error::message_size) {
//		LOG4CXX_INFO(logger, "Receiving " << len);
//		uint64_t* cmdPtr = (uint64_t*)rcvBuf.data();
//		uint64_t cmd = *cmdPtr;
//
//		if (cmd == 9) { //begin
//			LOG4CXX_INFO(logger, "To begin transfer");
//			uint64_t cmds[2];
//			cmds[0] = 10;
//			cmds[1] = 0;
//
//			sock.send(boost::asio::buffer((char*)cmds, sizeof(uint64_t) * 2));
//			rcv();
//		} else if (cmd == 10) { //transfer
//			uint64_t index = cmdPtr[1];
//			uint64_t tLen = cmdPtr[2];
//			LOG4CXX_INFO(logger, "Receive transfer " << index << ", " << tLen);
//
//			std::size_t bufLen = len - sizeof(uint64_t) * 3;
//			if (tLen != bufLen) {
//				LOG4CXX_ERROR(logger, "Error in  transfer " << tLen << " != " << bufLen);
//			}
//
//			char* cPtr = (char*)rcvBuf.data();
//			std::string str(cPtr + sizeof(uint64_t) * 3, tLen);
//			ss << str;
//
//			totalRcv += tLen;
//
//			uint64_t cmds[2];
//			cmds[0] = 10;
//			cmds[1] = totalRcv;
//			sock.send(boost::asio::buffer((char*)cmds, sizeof(uint64_t) * 2));
//			rcv();
//		} else if (cmd == 11) { //end
//			LOG4CXX_INFO(logger, "End of file");
//			try {
//				std::vector<torch::Tensor> ts;
//				torch::load(ts, ss);
//				LOG4CXX_INFO(logger, "ts size = " << ts.size());
//				for (const auto& t: ts) {
//					LOG4CXX_INFO(logger, "t: " << t.sizes());
//				}
//			} catch(std::exception& e) {
//				LOG4CXX_ERROR(logger, "Failed to convert tensor for " << e.what());
//			}
//
//			rcv();
//		} else {
//			LOG4CXX_INFO(logger, "unexpected cmd: " << cmd);
//		}
//	} else {
//		LOG4CXX_ERROR(logger, "Failed to receive message for: " << error);
//	}

	void* dataPtr = (void*)rcvBuf.data();
	rcvFunc(dataPtr, len);

//	rcv();
	peekRcv();
}

void A3CTCPServerConn::handleSnd(const boost::system::error_code& error, std::size_t len) {
	if (!error) {
//		LOG4CXX_INFO(logger, "Sent something " << len);
	} else {
		LOG4CXX_ERROR(logger, "Failure in receiving: " << error);
	}
}

void A3CTCPServerConn::handleRead(const boost::system::error_code& err) {
	if (!err) {
		boost::asio::async_read(sock, bs,
				boost::asio::transfer_at_least(1),
				boost::bind(&A3CTCPServerConn::handleRead, this->shared_from_this(),
						boost::asio::placeholders::error));
	} else if (err != boost::asio::error::eof) {
		LOG4CXX_INFO(logger, "end of reading ");
	} else {
		LOG4CXX_ERROR(logger, "Failed to read");
	}
}

bool A3CTCPServerConn::send(void* data, std::size_t len) {
	uint64_t* cmdPtr = (uint64_t*)data;
	uint64_t cmd = *cmdPtr;
	LOG4CXX_INFO(logger, "connection send cmd " << cmd);

    boost::asio::async_write(sock, boost::asio::buffer(data, len),
        boost::bind(&A3CTCPServerConn::handleSnd, this->shared_from_this(),
          boost::asio::placeholders::error,
          boost::asio::placeholders::bytes_transferred
		  ));
    return true;
//	auto sndLen = sock.send(boost::asio::buffer(data, len));
//	if (sndLen == len) {
//		LOG4CXX_INFO(logger, "sent something " << len);
//		return true;
//	} else {
//		LOG4CXX_ERROR(logger, "Not sent all data");
//		return false;
//	}
}
