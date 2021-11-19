/*
 * a3ctcpserverhdfacimpl.hpp
 *
 *  Created on: Nov 13, 2021
 *      Author: zf
 */

#ifndef INC_A3C_A3CTCPSERVERHDFACIMPL_HPP_
#define INC_A3C_A3CTCPSERVERHDFACIMPL_HPP_

#include "a3ctcpserverhandlefactory.h"
#include "a3ctcpserverhandle.hpp"

#include <memory>

//#define BOOST_ASIO_ENABLE_HANDLER_TRACKING
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/algorithm/string.hpp>

template<typename NetType, typename OptType>
class A3CTCPServerHdFacImpl: public A3CTCPHandleFactory {
public:
	virtual ~A3CTCPServerHdFacImpl();
	A3CTCPServerHdFacImpl(boost::asio::io_service& iio, NetType& iNet, OptType& iOpt);

	virtual std::shared_ptr<A3CTCPServerHandleInterface> createHandle();

private:
	boost::asio::io_service& ioService;
	NetType& net;
	OptType& opt;
};

template<typename NetType, typename OptType>
A3CTCPServerHdFacImpl<NetType, OptType>::~A3CTCPServerHdFacImpl() {

}

template<typename NetType, typename OptType>
A3CTCPServerHdFacImpl<NetType, OptType>::A3CTCPServerHdFacImpl(boost::asio::io_service& iio, NetType& iNet, OptType& iOpt)
	:ioService(iio),
	 net(iNet),
	 opt(iOpt)
{

}

template<typename NetType, typename OptType>
std::shared_ptr<A3CTCPServerHandleInterface> A3CTCPServerHdFacImpl<NetType, OptType>::createHandle() {
	return A3CTCPServerHandle<NetType, OptType>::CreateInterface(ioService, net, opt);
}

#endif /* INC_A3C_A3CTCPSERVERHDFACIMPL_HPP_ */
