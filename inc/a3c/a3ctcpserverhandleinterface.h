/*
 * a3ctcpserverhandleinterface.h
 *
 *  Created on: Nov 13, 2021
 *      Author: zf
 */
#ifndef INC_A3C_A3CTCPSERVERHANDLEINTERFACE_HPP_
#define INC_A3C_A3CTCPSERVERHANDLEINTERFACE_HPP_

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/algorithm/string.hpp>

class A3CTCPServerHandleInterface {
public:
	virtual ~A3CTCPServerHandleInterface() = 0;
	virtual boost::asio::ip::tcp::socket& getSock() = 0;
	virtual void start() = 0;
	virtual uint64_t getUpdateNum() = 0;
};

#endif
