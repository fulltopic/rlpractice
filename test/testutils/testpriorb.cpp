/*
 * testpriorb.cpp
 *
 *  Created on: Dec 31, 2021
 *      Author: zf
 */


#include "alg/utils/priorb.h"


#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/simplelayout.h>
#include <log4cxx/logmanager.h>


namespace {
void test0() {
	at::IntArrayRef inputShape{2, 2};
	const int cap = 8;
	const float epsilon = 1e-6;
	PrioReplayBuffer buffer(cap, inputShape, epsilon);

	const int batchSize = cap / 2;
	for (int i = 0; i < cap; i ++) {
		torch::Tensor state = torch::ones({2});
		torch::Tensor nextState = torch::ones({2});
//		std::cout << state << std::endl;
//		std::cout << nextState << std::endl;

		buffer.add(state, nextState, 0, 0, 0);
	}
	std::cout << "After full fill" << std::endl;
//	buffer.print();


	for (int i = 0; i < cap; i ++) {
		auto state = torch::ones({2});
		auto nextState = torch::ones({2});
		buffer.add(state, nextState, 0, 0, 0);
		std::cout << "----------------------------> After add " << i << std::endl;
//		buffer.print();

		auto samples = buffer.getSampleIndex(batchSize);
		auto index = std::get<0>(samples);
		auto prios = std::get<1>(samples);
//		std::cout << "samples " << std::endl;
//		std::cout << "index: " << std::endl << index << std::endl;
//		std::cout << "prios: " << std::endl << prios << std::endl;

//		std::cout << "after sample " << i << std::endl;
//		buffer.print();

		auto newPrio = torch::randn(prios.sizes()).abs();
//		std::cout << "newe prios " << std::endl << newPrio << std::endl;

		buffer.update(index, newPrio);
//		std::cout << "after update " << i << std::endl;
//		buffer.print();
	}
}

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

	test0();

	return 0;
}

