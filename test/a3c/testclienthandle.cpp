/*
 * testclienthandle.cpp
 *
 *  Created on: Nov 12, 2021
 *      Author: zf
 */


#include "a3c/a3ctcpclienthanle.hpp"
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
void testCompile() {
	boost::asio::io_service iio;

	CartACFcNet net(12, 2);
	auto client = A3CTCPClientHandle<CartACFcNet>::Create(iio, net);

	client->start();
}

void testConn() {
	boost::asio::io_service iio;

	CartACFcNet net(12, 2);
	auto client = A3CTCPClientHandle<CartACFcNet>::Create(iio, net);
	client->start();

	std::unique_ptr<std::thread> t = std::make_unique<std::thread>(
		static_cast<std::size_t (boost::asio::io_context::*) ()>(&boost::asio::io_context::run), &iio);


	client->sendTest();

	t->join();
}

void testTarget() {
	boost::asio::io_service iio;

//	CartACFcNet net(12, 2);
	AirACCnnNet net(9);
	auto client = A3CTCPClientHandle<AirACCnnNet>::Create(iio, net);
	client->start();

	std::unique_ptr<std::thread> t = std::make_unique<std::thread>(
		static_cast<std::size_t (boost::asio::io_context::*) ()>(&boost::asio::io_context::run), &iio);


	client->syncTarget();

	t->join();
}

void testGrad() {
	const int inputNum = 12;
	const int batchSize = 2;
	const int outputNum = 2;

	boost::asio::io_service iio;

	CartACFcNet net(inputNum, outputNum);
//	AirACCnnNet net(9);
	auto client = A3CTCPClientHandle<CartACFcNet>::Create(iio, net);
	client->start();

	std::unique_ptr<std::thread> t = std::make_unique<std::thread>(
		static_cast<std::size_t (boost::asio::io_context::*) ()>(&boost::asio::io_context::run), &iio);


	torch::Tensor input = torch::rand({batchSize, inputNum});
	torch::Tensor valueTarget = torch::rand({batchSize, 1});
	torch::Tensor actionTarget = torch::rand({batchSize, outputNum});

	auto output = net.forward(input);
	torch::Tensor valueOutput = output[1];
	torch::Tensor actionOutput = output[0];

	torch::Tensor valueLoss =  torch::nn::functional::mse_loss(valueTarget, valueOutput);
	torch::Tensor actionLoss =  torch::nn::functional::huber_loss(actionTarget, actionOutput);
	torch::Tensor loss = valueLoss + actionLoss;
	loss.backward();

	std::vector<torch::Tensor> grads;
	std::vector<torch::Tensor> params = net.parameters();
	for (int i = 0; i < params.size(); i ++) {
		grads.push_back(params[i].grad());
	}
	client->addGrad(grads);

	client->sendGrad();

	t->join();
}


void testAirGrad() {
	const int inputNum = 12;
	const int batchSize = 2;
	const int outputNum = 9;

	boost::asio::io_service iio;

	AirACCnnNet net(9);
	auto client = A3CTCPClientHandle<AirACCnnNet>::Create(iio, net);
	client->start();

	std::unique_ptr<std::thread> t = std::make_unique<std::thread>(
		static_cast<std::size_t (boost::asio::io_context::*) ()>(&boost::asio::io_context::run), &iio);


	torch::Tensor input = torch::rand({batchSize, 4, 84, 84});
	torch::Tensor valueTarget = torch::rand({batchSize, 1});
	torch::Tensor actionTarget = torch::rand({batchSize, outputNum});

	auto output = net.forward(input);
	torch::Tensor valueOutput = output[1];
	torch::Tensor actionOutput = output[0];

	torch::Tensor valueLoss =  torch::nn::functional::mse_loss(valueTarget, valueOutput);
	torch::Tensor actionLoss =  torch::nn::functional::huber_loss(actionTarget, actionOutput);
	torch::Tensor loss = valueLoss + actionLoss;
	loss.backward();

	std::vector<torch::Tensor> grads;
	std::vector<torch::Tensor> params = net.parameters();
	for (int i = 0; i < params.size(); i ++) {
		grads.push_back(params[i].grad());
	}
	client->addGrad(grads);

	client->sendGrad();

	t->join();
}

void testLoad() {
	std::vector<torch::Tensor> ts;
	torch::load(ts, "./test.pt");
	std::cout << "ts: " << ts.size() << std::endl;
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

//	testConn();
	testTarget();
//	testGrad();
//	testAirGrad();

//	testLoad();

	return 0;
}
