/*
 * a3cclient.cpp
 *
 *  Created on: Nov 5, 2021
 *      Author: zf
 */

#ifndef SRC_A3C_A3CCLIENT_CPP_
#define SRC_A3C_A3CCLIENT_CPP_

#include "a3c/a3cclient.h"

#include <sstream>

std::shared_ptr<A3CClient> A3CClient::Create(const std::string ip, const int port, int seqNum, boost::asio::io_context& iio) {
//	return std::make_shared<A3CClient>(ip, port, seqNum, iio); //TODO: private access, why?
	return std::shared_ptr<A3CClient>(new A3CClient(ip, port, seqNum, iio));
}

A3CClient::A3CClient(const std::string ip, const int port, int seqNum, boost::asio::io_context& iio):
	serverIp(ip),
	serverPort(port),
	serverP(boost::asio::ip::address::from_string(ip), port),
	seq(seqNum),
	resolver(iio),
	sock(iio)
{
	sock.open(boost::asio::ip::udp::udp::v4());

//	startRcv();
}

A3CClient::~A3CClient() {
	sock.close();
}

void A3CClient::startRcv() {
	sock.async_receive_from( boost::asio::buffer(rcvBuf), remoteEp,
			boost::bind(&A3CClient::handleRcv, this->shared_from_this(),
					boost::asio::placeholders::error,
					boost::asio::placeholders::bytes_transferred));
}

void A3CClient::handleRcv(const boost::system::error_code& error, std::size_t len) {
	LOG4CXX_INFO(logger, "nothing to handle");

	startRcv();
}

bool A3CClient::send(torch::Tensor tensor) {
//	torch::serialize::OutputArchive outputArchive;
//	tensor.save(outputArchive);
	std::vector<torch::Tensor> ts(2);
	ts[0] = torch::zeros({4, 4});
	ts[1] = torch::ones({3, 3});
	std::ostringstream os;
	torch::save(ts, os);
//	outputArchive.save_to(os);

//	os.rdbuf();
	std::string strBuf = os.str();
	LOG4CXX_INFO(logger, "Send to " << serverP);

	std::ostringstream os1;
	std::ostringstream os2;
	torch::save(ts[0], os1);
	torch::save(ts[1], os1);
	LOG4CXX_INFO(logger, "size 0: " << os1.str().length());
//	LOG4CXX_INFO(logger, "size 1: " << os2.str().length());


	std:size_t expLen = strBuf.length();
	sock.async_send_to(boost::asio::buffer(strBuf.data(), expLen), serverP,
			boost::bind(&A3CClient::handleSend, this->shared_from_this(),
					boost::asio::placeholders::error,
					boost::asio::placeholders::bytes_transferred,
					expLen));
	return true;
}

bool A3CClient::send(std::vector<torch::Tensor> ts) {
	std::ostringstream os;
	torch::save(ts, os);
	std::string strBuf = os.str();

	std::size_t index = 0;

	while (index < strBuf.length()) {
		try {
			auto bufLen = std::min(strBuf.length() - index, (std::size_t)A3CConfig::BufCap);
			LOG4CXX_INFO(logger, "bufLen = " << bufLen);
			auto sentBytes = sock.send_to(boost::asio::buffer(strBuf.data() + index, bufLen), serverP);
			LOG4CXX_INFO(logger, "Sent: " << strBuf.length() << " to " << serverP );
			index += bufLen;
//			return true;
		} catch (boost::system::system_error& e) {
			LOG4CXX_ERROR(logger, "Sent error " << seq << ", " << e.what());
			return false;
		}
	}
	LOG4CXX_INFO(logger, "Sent: " << index);
	return true;
}


void A3CClient::handleSend(const boost::system::error_code& e, std::size_t len, std::size_t expLen) {
	LOG4CXX_INFO(logger, "handleSend");
	if ((!e) || (e == boost::asio::error::message_size)) {
//		logger->info("Client{}:{} push buffer {}: {}?{}", room->seq, index, bufIndex, len, expLen);
		if (len > 0) {
			LOG4CXX_INFO(logger, "Sent " << len << ": " << expLen);
		}
	} else {
		LOG4CXX_ERROR(logger, "Client" << seq << " sending failure " << e.message());
	}
}

#endif /* SRC_A3C_A3CCLIENT_CPP_ */
