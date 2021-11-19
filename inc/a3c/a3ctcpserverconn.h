/*
 * a3ctcpserverconn.h
 *
 *  Created on: Nov 9, 2021
 *      Author: zf
 */

#ifndef INC_A3C_A3CTCPSERVERCONN_H_
#define INC_A3C_A3CTCPSERVERCONN_H_

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
#include <sstream>
#include <functional>

#include "a3ctcpconfig.h"
#include "dummyfuncs.h"

class A3CTCPServerConn: public std::enable_shared_from_this<A3CTCPServerConn> {
public:
	~A3CTCPServerConn() = default;
	A3CTCPServerConn(const A3CTCPServerConn&) = delete;

	static std::shared_ptr<A3CTCPServerConn> Create(boost::asio::io_service& iio);
	void start();

	boost::asio::ip::tcp::socket& getSock();
	void setRcvFunc(std::function<void(void*, std::size_t len)> func);
	bool send(void* dataPtr, std::size_t len);
private:
	A3CTCPServerConn(boost::asio::io_service& iio);

	boost::asio::ip::tcp::socket sock;
	boost::array<char, A3CTCPConfig::BufCap> rcvBuf; //TODO: buffer size to be extended
	boost::asio::ip::tcp::endpoint remoteEp;

	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("a3cs");
	std::stringstream ss;
	boost::asio::streambuf bs;
	std::size_t totalRcv = 0;

	std::function<void(void*, std::size_t)> rcvFunc;

	void peekRcv();
	void handlePeekRcv(const boost::system::error_code& error, std::size_t);

//	void rcv();
	void handleRcv(const boost::system::error_code& error, std::size_t);
	void handleSnd(const boost::system::error_code& error, std::size_t);
	void handleRead(const boost::system::error_code& error);
};


#endif /* INC_A3C_A3CTCPSERVERCONN_H_ */
