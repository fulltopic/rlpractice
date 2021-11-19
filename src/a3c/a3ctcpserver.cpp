/*
 * a3ctcpserver.cpp
 *
 *  Created on: Nov 10, 2021
 *      Author: zf
 */


#include "a3c/a3ctcpconfig.h"
#include "a3c/a3ctcpserver.h"
#include "a3c/a3ctcpserverconn.h"

std::shared_ptr<A3CTCPServer> A3CTCPServer::Create(boost::asio::io_service& iio, std::shared_ptr<A3CTCPHandleFactory> iFactory)
{
	return std::shared_ptr<A3CTCPServer>(new A3CTCPServer(iio, iFactory));
}

A3CTCPServer::A3CTCPServer(boost::asio::io_service& iio, std::shared_ptr<A3CTCPHandleFactory> iFactory):
		ioService(iio),
		factory(iFactory),
	acceptor(iio, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), A3CTCPConfig::ServerPort))
{
}

void A3CTCPServer::startAccept() {
//	std::shared_ptr<A3CTCPServerConn> conn = A3CTCPServerConn::Create(ioService);
	auto handle = factory->createHandle();
	clients.push_back(handle);

//	acceptor.async_accept(conn->getSock(),
//			boost::bind(&A3CTCPServer::handleAccept, this->shared_from_this(), conn,
//					boost::asio::placeholders::error));


	acceptor.async_accept(handle->getSock(),
			boost::bind(&A3CTCPServer::handleAccept, this->shared_from_this(), handle,
					boost::asio::placeholders::error));
}

void A3CTCPServer::handleAccept(std::shared_ptr<A3CTCPServerHandleInterface> conn, const boost::system::error_code& error) {
	if (!error) {
		LOG4CXX_INFO(logger, "accept a connection");
		conn->start();
	} else {
		LOG4CXX_INFO(logger, "failed to accept connection as: " << error);
	}

	startAccept();
}
