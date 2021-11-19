/*
 * a3cserver.cpp
 *
 *  Created on: Nov 5, 2021
 *      Author: zf
 */

#include "a3c/a3cserver.h"

#include <sstream>

std::shared_ptr<A3CServer> A3CServer::Create(const std::string ip, const int port, boost::asio::io_context& iio, A3CClientHandler& handler) {
	return std::shared_ptr<A3CServer>(new A3CServer(ip, port, iio, handler));
}

A3CServer::A3CServer(const std::string ip, const int port, boost::asio::io_context& iio, A3CClientHandler& handler):
	serverIp(ip),
	serverPort(port),
	serverP(boost::asio::ip::address::from_string(serverIp), serverPort),
	resolver(iio),
	sock(iio, boost::asio::ip::udp::endpoint(boost::asio::ip::address::from_string(serverIp), serverPort)),
	handle(handler)
{
//	startRcv();
}

A3CServer::~A3CServer() {
	sock.close();
}

void A3CServer::startRcv() {
	sock.async_receive_from( boost::asio::buffer(rcvBuf), remoteEp,
			boost::bind(&A3CServer::rcv, this->shared_from_this(),
					boost::asio::placeholders::error,
					boost::asio::placeholders::bytes_transferred));
}

void A3CServer::rcv(const boost::system::error_code& error, std::size_t len) {
	LOG4CXX_INFO(logger, "Receive from " << remoteEp << " len = " << len);

//    if (!error || error == boost::asio::error::message_size) {
//    	std::string data(rcvBuf.data(), len);
//    	std::istringstream iss(data);
//    	std::vector<torch::Tensor> ts;
//    	torch::load(ts, iss);
//    	LOG4CXX_INFO(logger, "ts size: " << ts.size());
//    	LOG4CXX_INFO(logger, "cmd: " << ts[0]);
//    	for (const auto& t: ts) {
//    		LOG4CXX_INFO(logger, "t: " << t.sizes());
//    	}
////    	LOG4CXX_INFO(logger, "received: " << tensor);
//    } else {
//    	LOG4CXX_ERROR(logger, "Failed to receive message for " << error);
//    }
	if (!error || error == boost::asio::error::message_size) {
		if (len >= A3CConfig::BufCap) {
			totalRcv += len;
			LOG4CXX_INFO(logger, "Received: " << len << ", " << totalRcv);
			std::string data(rcvBuf.data(), len);
			is << data;

			startRcv();
		} else {
			totalRcv += len;
			LOG4CXX_INFO(logger, "Received last: " << len << ", " << totalRcv);
			std::string data(rcvBuf.data(), len);
			is << data;
			LOG4CXX_INFO(logger, "End of rcv");
			std::vector<torch::Tensor> ts;
			torch::load(ts, is);
			LOG4CXX_INFO(logger, "ts size: " << ts.size());
			for (const auto& t: ts) {
				LOG4CXX_INFO(logger, "t: " << t.sizes());
			}

			totalRcv = 0;
		}
	} else {
		LOG4CXX_ERROR(logger, "Failed to receive message for: " << error);
	}
}
