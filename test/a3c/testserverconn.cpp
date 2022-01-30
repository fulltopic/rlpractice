/*
 * testserverconn.cpp
 *
 *  Created on: Nov 10, 2021
 *      Author: zf
 */



#include "a3c/a3ctcpserver.h"
#include "a3c/a3ctcpconfig.h"
#include "gymtest/cnnnets/lunarnets/cartacnet.h"

#include <iostream>
#include <string>


#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/simplelayout.h>
#include <log4cxx/logmanager.h>

namespace {
//void testRcv() {
//		CartACFcNet net(12, 2);
////		AirACCnnNet net(9);
//		std::vector<at::IntArrayRef> shapes;
//		auto params = net.parameters();
//		for (const auto& param: params) {
//			shapes.push_back(param.sizes());
//		}
//
//		for(int i = 0; i < shapes.size(); i ++) {
//			std::cout << "shapes " << i << " = " << shapes[i] << std::endl;
//		}
//
//		A3CClientHandler clientHandler(shapes);
//
//
//	boost::asio::io_service iio;
//	std::shared_ptr<A3CServer> server = A3CServer::Create(A3CConfig::ServerIp, A3CConfig::ServerPort, iio, clientHandler);
//	server->startRcv();
//
//	std::unique_ptr<std::thread> t = std::make_unique<std::thread>(
//			static_cast<std::size_t (boost::asio::io_context::*) ()>(&boost::asio::io_context::run), &iio);
//	t->join();
//}

void testRcv() {
//	CartACFcNet net(12, 2);
//
//	boost::asio::io_service iio;
//	std::shared_ptr<A3CTCPServer> server = A3CTCPServer::Create(iio);
//	server->startAccept();
//
//	std::unique_ptr<std::thread> t = std::make_unique<std::thread>(
//		static_cast<std::size_t (boost::asio::io_context::*) ()>(&boost::asio::io_context::run), &iio);
//	t->join();
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

	testRcv();

	return 0;
}


