/*
 * a3ctcpserverqhdfacimpl.hpp
 *
 *  Created on: Dec 2, 2021
 *      Author: zf
 */

#ifndef INC_A3C_A3CTCPSERVERQHDFACIMPL_HPP_
#define INC_A3C_A3CTCPSERVERQHDFACIMPL_HPP_


#include "a3ctcpserverhandlefactory.h"
#include "a3ctcpserverqhdl.hpp"

#include <memory>

//#define BOOST_ASIO_ENABLE_HANDLER_TRACKING
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/algorithm/string.hpp>

//TODO: Move facimpl into handle itself.
template<typename NetType>
class A3CTCPServerQHdFacImpl: public A3CTCPHandleFactory {
public:
	virtual ~A3CTCPServerQHdFacImpl();
	A3CTCPServerQHdFacImpl(boost::asio::io_service& iio, NetType& iNet, A3CGradQueue& iQ);

	virtual std::shared_ptr<A3CTCPServerHandleInterface> createHandle();

private:
	boost::asio::io_service& ioService;
	NetType& net;
	A3CGradQueue& q;
};

template<typename NetType>
A3CTCPServerQHdFacImpl<NetType>::~A3CTCPServerQHdFacImpl() {

}

template<typename NetType>
A3CTCPServerQHdFacImpl<NetType>::A3CTCPServerQHdFacImpl(boost::asio::io_service& iio, NetType& iNet, A3CGradQueue& iQ)
	:ioService(iio),
	 net(iNet),
	 q(iQ)
{

}

template<typename NetType>
std::shared_ptr<A3CTCPServerHandleInterface> A3CTCPServerQHdFacImpl<NetType>::createHandle() {
	return A3CTCPServerQHandle<NetType>::CreateInterface(ioService, net, q);
}


#endif /* INC_A3C_A3CTCPSERVERQHDFACIMPL_HPP_ */
