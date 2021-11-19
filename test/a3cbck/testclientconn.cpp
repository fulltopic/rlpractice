/*
 * testclientconn.cpp
 *
 *  Created on: Nov 5, 2021
 *      Author: zf
 */

#include "a3c/a3cclient.h"
#include "a3c/a3cconfig.h"

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

namespace {
void testSend() {
	A3CConfig config;
	boost::asio::io_service iio;
	std::shared_ptr<A3CClient> client = A3CClient::Create(A3CConfig::ServerIp, A3CConfig::ServerPort, 1, iio);
	client->startRcv();

	std::unique_ptr<std::thread> t = std::make_unique<std::thread>(
			static_cast<std::size_t (boost::asio::io_context::*) ()>(&boost::asio::io_context::run), &iio);
	torch::Tensor tensor = torch::ones({4, 4});
	client->send(tensor);

	t->join();
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

	testSend();

	return 0;
}
