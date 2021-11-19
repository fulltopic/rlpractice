/*
 * testserverhandle.cpp
 *
 *  Created on: Nov 12, 2021
 *      Author: zf
 */




#include "a3c/a3ctcpserverhandle.hpp"
#include "a3c/a3ctcpserverconn.h"
#include "a3c/a3ctcpconfig.h"
#include "a3c/a3ctcpserverhdfacimpl.hpp"
#include "a3c/a3ctcpserver.h"

#include <iostream>
#include <string>
#include <memory>
#include <thread>


#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/simplelayout.h>
#include <log4cxx/logmanager.h>

#include "gymtest/lunarnets/cartacnet.h"
#include "gymtest/airnets/airacnet.h"

namespace {
void testCompile() {
	boost::asio::io_service iio;

	CartACFcNet net(12, 2);
    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-2));

//    auto a = new A3CTCPServerHdFacImpl<CartACFcNet, torch::optim::Adam> >(iio, net, optimizer)

    std::shared_ptr<A3CTCPHandleFactory> factory
		= std::shared_ptr<A3CTCPHandleFactory>(
				new A3CTCPServerHdFacImpl<CartACFcNet, torch::optim::Adam>(iio, net, optimizer)
				);

    std::shared_ptr<A3CTCPServer> server = A3CTCPServer::Create(iio, factory);

//	auto client = A3CTCPServerHandle<CartACFcNet, torch::optim::Adam>::Create(iio, net, optimizer);
//
//	client->start();
}

void testConn() {
//	CartACFcNet net(12, 2);
	AirACCnnNet net(9);

    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-2));
	boost::asio::io_service iio;

    std::shared_ptr<A3CTCPHandleFactory> factory
		= std::shared_ptr<A3CTCPHandleFactory>(
				new A3CTCPServerHdFacImpl<AirACCnnNet, torch::optim::Adam>(iio, net, optimizer)
				);

	std::shared_ptr<A3CTCPServer> server = A3CTCPServer::Create(iio,factory);
	server->startAccept();

	std::unique_ptr<std::thread> t = std::make_unique<std::thread>(
		static_cast<std::size_t (boost::asio::io_context::*) ()>(&boost::asio::io_context::run), &iio);


	t->join();
}

void testSave() {
	AirACCnnNet net(9);
	std::stringstream target;
	target.clear();
	std::vector<torch::Tensor> params = net.parameters();
	torch::save(params, target);
}
}


namespace {
void logConfigure(bool err) {
    log4cxx::ConsoleAppenderPtr appender(new log4cxx::ConsoleAppender());
    if (err) {
        appender->setTarget(LOG4CXX_STR("System.err"));
    }
    log4cxx::LayoutPtr layout(new log4cxx::SimpleLayout());
    appender->setLayout(layout);
    log4cxx::helpers::Pool pool;
    appender->activateOptions(pool);
    log4cxx::Logger::getRootLogger()->addAppender(appender);
//    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getDebug());
    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getInfo());
//    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getError());
    log4cxx::LogManager::getLoggerRepository()->setConfigured(true);
}
}

int main() {
	logConfigure(false);

//	testSend();
//	testSendTensor();
//	testCompile();
	testConn();

//	testSave();

	return 0;
}
