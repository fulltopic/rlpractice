/*
 * a3cserver.h
 *
 *  Created on: Nov 9, 2021
 *      Author: zf
 */

#ifndef INC_A3C_A3CSERVER_H_
#define INC_A3C_A3CSERVER_H_

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
#include <atomic>

#include <torch/torch.h>

#include "a3c/a3ctcpserverconn.h"
#include "a3ctcpserverhandlefactory.h"
#include "a3ctcpserverhandleinterface.h"

class A3CTCPServer: public std::enable_shared_from_this<A3CTCPServer> {
public:
	~A3CTCPServer() = default;
	A3CTCPServer(const A3CTCPServer&) = delete;

	static std::shared_ptr<A3CTCPServer> Create(boost::asio::io_service& iio, std::shared_ptr<A3CTCPHandleFactory> iFactory);

	void start();
//	void handleAccept(std::shared_ptr<A3CTCPServerConn> conn, const boost::system::error_code& error);
	void handleAccept(std::shared_ptr<A3CTCPServerHandleInterface> conn, const boost::system::error_code& error);

	uint64_t getUpdateNum();
	void setPollMinute(int minute);
private:
	A3CTCPServer(boost::asio::io_service& iio, std::shared_ptr<A3CTCPHandleFactory> iFactory);

	std::shared_ptr<A3CTCPHandleFactory> factory;
	boost::asio::ip::tcp::acceptor acceptor;

	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("a3cserver");
	boost::asio::io_service& ioService;

	std::vector<std::shared_ptr<A3CTCPServerHandleInterface> > clients;

	volatile uint64_t updateNum;
	int pollMinute = 10;
	boost::asio::deadline_timer pollTimer;

	void startAccept();


	void handlePoll(const boost::system::error_code& error);
	void pollConns();
};



#endif /* INC_A3C_A3CSERVER_H_ */
