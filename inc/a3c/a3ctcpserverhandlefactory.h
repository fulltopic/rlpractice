/*
 * a3ctcphandlefactory.h
 *
 *  Created on: Nov 13, 2021
 *      Author: zf
 */
#ifndef INC_A3C_A3CTCPSERVERHANDLEFACTORY_HPP_
#define INC_A3C_A3CTCPSERVERHANDLEFACTORY_HPP_

#include "a3ctcpserverhandleinterface.h"

#include <memory>

class A3CTCPHandleFactory {
public:
	virtual ~A3CTCPHandleFactory() = 0;
	virtual std::shared_ptr<A3CTCPServerHandleInterface> createHandle() = 0;
};


#endif
