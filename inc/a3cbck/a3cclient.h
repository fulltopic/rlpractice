/*
 * a3cclient.h
 *
 *  Created on: Nov 5, 2021
 *      Author: zf
 */

#ifndef INC_A3C_A3CCLIENT_H_
#define INC_A3C_A3CCLIENT_H_

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

#include "a3cconfig.h"
#include "commdata.h"

using boost::asio::ip::tcp;
using ErrorCode=boost::system::error_code;

class A3CClient
		:public std::enable_shared_from_this<A3CClient>{

private:
	const std::string serverIp;
	const int serverPort;
	const int seq;

	boost::asio::ip::udp::resolver resolver;
	boost::asio::ip::udp::endpoint serverP;
	boost::asio::ip::udp::socket sock;
	boost::array<char, A3CConfig::BufCap> rcvBuf; //TODO: buffer size to be extended
	boost::asio::ip::udp::endpoint remoteEp;

	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("a3cc");
	void handleSend(const boost::system::error_code& e, std::size_t len, std::size_t expLen);
	void handleRcv(const boost::system::error_code& error, std::size_t len);
	A3CClient(const std::string ip, const int port, int seqNum, boost::asio::io_context& iio);


public:
	~A3CClient();
	A3CClient(const A3CClient&) = delete;

	bool send(torch::Tensor tensor);
	bool send(std::vector<torch::Tensor> datas);
	void startRcv();

	static std::shared_ptr<A3CClient> Create(const std::string ip, const int port, int seqNum, boost::asio::io_context& iio);
};



#endif /* INC_A3C_A3CCLIENT_H_ */
