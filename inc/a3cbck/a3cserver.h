/*
 * a3cserver.h
 *
 *  Created on: Nov 5, 2021
 *      Author: zf
 */

#ifndef INC_A3C_A3CSERVER_H_
#define INC_A3C_A3CSERVER_H_

#include "a3cconfig.h"

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
#include "a3cclienthandle.h"

using boost::asio::ip::udp;
using ErrorCode=boost::system::error_code;

class A3CServer
		:public std::enable_shared_from_this<A3CServer> {
private:
	const std::string serverIp;
	const int serverPort;

	boost::asio::ip::udp::resolver resolver;
	boost::asio::ip::udp::endpoint serverP;
	boost::asio::ip::udp::socket sock;
	boost::array<char, A3CConfig::BufCap> rcvBuf; //TODO: buffer size to be extended
	boost::asio::ip::udp::endpoint remoteEp;

	A3CClientHandler& handle;
	std::stringstream is;
	std::size_t totalRcv = 0;

	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("a3cs");

	void rcv(const boost::system::error_code& error, std::size_t);

	A3CServer(const std::string ip, const int port, boost::asio::io_context& iio, A3CClientHandler& handler);

public:
	~A3CServer();
	A3CServer(const A3CServer&) = delete;

	void startRcv();

	static std::shared_ptr<A3CServer> Create(const std::string ip, const int port, boost::asio::io_context& iio, A3CClientHandler& handler);
//	bool send(torch::Tensor tensor);
};



#endif /* INC_A3C_A3CSERVER_H_ */
