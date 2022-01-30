/*
 * testclientconn.cpp
 *
 *  Created on: Nov 10, 2021
 *      Author: zf
 */





#include "a3c/a3ctcpclientconn.h"
#include "a3c/a3ctcpconfig.h"

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

#include "gymtest/cnnnets/lunarnets/cartacnet.h"
#include "gymtest/cnnnets/airnets/airacnet.h"
namespace {
void testSend() {
	A3CTCPConfig config;
	boost::asio::io_service iio;
	std::shared_ptr<A3CTCPClientConn> client = A3CTCPClientConn::Create(iio);
	client->start();

	std::unique_ptr<std::thread> t = std::make_unique<std::thread>(
			static_cast<std::size_t (boost::asio::io_context::*) ()>(&boost::asio::io_context::run), &iio);
	torch::Tensor tensor = torch::ones({4, 4});
	client->send({tensor});

	t->join();
}

void testSendTensor() {
	boost::asio::io_service iio;
	std::shared_ptr<A3CTCPClientConn> client = A3CTCPClientConn::Create(iio);
	client->start();

	std::unique_ptr<std::thread> t = std::make_unique<std::thread>(
			static_cast<std::size_t (boost::asio::io_context::*) ()>(&boost::asio::io_context::run), &iio);

//	CartACFcNet net(12, 2);
	AirACCnnNet net(9);
	auto params = net.parameters();
	for(int i = 0; i < params.size(); i ++) {
		std::cout << "shapes " << i << " = " << params[i].sizes() << std::endl;
	}
	client->send(params);

	t->join();
}

void testGrad() {
	const int actionNum = 9;
	const int batchSize = 2;

	AirACCnnNet net(actionNum);
    torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));

    torch::Tensor valueTarget = torch::rand({batchSize});
    torch::Tensor input = torch::rand({batchSize, 4, 84, 84});

    auto output = net.forward(input);
    torch::Tensor valueOutput = output[1].view({batchSize});
    auto valueLoss = torch::nn::functional::huber_loss(valueOutput, valueTarget);
    valueLoss.backward();

    auto params = net.parameters();
    std::vector<torch::Tensor> grads;
    for (int i = 0; i < params.size(); i ++) {
    	torch::Tensor gradTensor = params[i].grad();
    	if (!params[i].requires_grad()) {
    		std::cout << "grad " << i << " has no grad " << std::endl;
    	}
    	if (!gradTensor.defined()){
    		std::cout << "grad " << i << " not defined " << std::endl;
    	}
    	if (gradTensor.numel() == 0) {
    		grads.push_back(torch::zeros({0}));
    	} else {
    		grads.push_back(gradTensor);
    	}
    	std::cout << "grad " << i << " = " << gradTensor.sizes() << std::endl;
    }

	boost::asio::io_service iio;
	std::shared_ptr<A3CTCPClientConn> client = A3CTCPClientConn::Create(iio);
	client->start();

	std::unique_ptr<std::thread> t = std::make_unique<std::thread>(
			static_cast<std::size_t (boost::asio::io_context::*) ()>(&boost::asio::io_context::run), &iio);

	client->send(grads);

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

//	testSend();
//	testSendTensor();
	testGrad();

	return 0;
}
