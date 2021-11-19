/*
 * a3ctcpclientconn.h
 *
 *  Created on: Nov 9, 2021
 *      Author: zf
 */

#ifndef INC_A3C_A3CTCPCLIENTCONN_H_
#define INC_A3C_A3CTCPCLIENTCONN_H_

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

#include <torch/torch.h>

#include "a3c/a3ctcpconfig.h"
#include "a3c/dummyfuncs.h"

class A3CTCPClientConn:
	public std::enable_shared_from_this<A3CTCPClientConn> {
public:
	~A3CTCPClientConn();

	static std::shared_ptr<A3CTCPClientConn> Create(boost::asio::io_service& iio);

	void setRcvFunc(std::function<void(void*, std::size_t)> func);

	void start();
	bool send(std::vector<torch::Tensor> ts);
	bool send(void* data, std::size_t len);

private:
	A3CTCPClientConn(boost::asio::io_service& iio);

	void peekRcv();
	void handlePeekRcv(const boost::system::error_code& error, std::size_t len);

//	void rcv();
	void handleRcv(const boost::system::error_code& error, std::size_t len);
	void handleSnd(const boost::system::error_code& error, std::size_t len);

	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("a3cclientconn");
	boost::asio::ip::tcp::resolver resolver;
	boost::asio::ip::tcp::endpoint serverP;
	boost::asio::ip::tcp::socket sock;
	boost::array<char, A3CTCPConfig::BufCap> rcvBuf; //TODO: buffer size to be extended
	boost::asio::ip::tcp::endpoint remoteEp;

	std::string tmpFileStr;
	std::ostringstream tmpFileSs;

//	DummyFuncs dummy;
	std::function<void(void*, std::size_t)> rcvFunc;
};



#endif /* INC_A3C_A3CTCPCLIENTCONN_H_ */
