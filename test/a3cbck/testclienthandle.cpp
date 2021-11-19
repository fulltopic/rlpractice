/*
 * testclienthandle.cpp
 *
 *  Created on: Nov 8, 2021
 *      Author: zf
 */



#include "a3c/a3cclient.h"
#include "a3c/a3cconfig.h"
#include "a3c/a3cclienthandle.h"

#include "gymtest/lunarnets/cartacnet.h"
#include "gymtest/airnets/airacnet.h"

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
void testInit() {
	boost::asio::io_service iio;
	std::shared_ptr<A3CClient> client = A3CClient::Create(A3CConfig::ServerIp, A3CConfig::ServerPort, 1, iio);

//	CartACFcNet net(12, 2);
	AirACCnnNet net(9);
	std::vector<at::IntArrayRef> shapes;
	auto params = net.parameters();
	for (const auto& param: params) {
		shapes.push_back(param.sizes());
	}

	for(int i = 0; i < shapes.size(); i ++) {
		std::cout << "shapes " << i << " = " << shapes[i] << std::endl;
	}

	A3CClientHandler clientHandler(shapes);
	clientHandler.setClient(client);

	clientHandler.sendGrads();
}

void testGrad() {

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
    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getDebug());
//    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getInfo());
//    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getError());
    log4cxx::LogManager::getLoggerRepository()->setConfigured(true);
}
}

int main() {
	logConfigure(false);

	testInit();

	return 0;
}

