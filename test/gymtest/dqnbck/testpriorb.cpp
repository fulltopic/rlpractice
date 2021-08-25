/*
 * testpriorb.cpp
 *
 *  Created on: May 24, 2021
 *      Author: zf
 */



#include "gymtest/utils/priorb.h"



#include <torch/torch.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/simplelayout.h>
#include <log4cxx/logmanager.h>

#include <vector>
#include <string>
#include <iostream>

namespace {
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("prtest"));

void testSum() {
	const int capacity = 8;
	PrioRb rb(capacity);

	for (int i = 0; i < capacity / 2; i ++) {
		rb.put({i}, {i * 1.0}, {i}, {i}, {false}, 1);
	}
	std::cout << "Put half" << std::endl;
	std::cout << "half sum = " << rb.getSum() << std::endl;

	torch::Tensor halfPrios = torch::rand({capacity / 2});
	std::cout << "update half: " << std::endl << halfPrios << std::endl;
	std::cout << "exp sum = " << halfPrios.sum() << std::endl;
	torch::Tensor halfDist = halfPrios / halfPrios.sum();
	std::cout << "exp dist = " << halfDist << std::endl;

	float* hp = halfPrios.data_ptr<float>();
	for (int i = 0; i < capacity / 2; i ++) {
		rb.updatePrios({i}, {hp[i]});
	}
	std::cout << "sum = " << rb.getSum() << std::endl;

	std::vector<int> hc(capacity, 0);
	const int heNum = 100;
	const int hbSize = capacity / 2;
	for (int i = 0; i < heNum; i ++) {
		auto rc = rb.sampleBatch(hbSize);
		auto idx = std::get<0>(rc);
//		LOG4CXX_DEBUG(logger, "sample idx " << idx);
		for (const auto& id: idx) {
			hc[id] ++;
		}
	}
	LOG4CXX_INFO(logger, "sample nums " << hc);
	std::vector<float> hcDist(hc.size(), 0);
	for (int i = 0; i < hc.size(); i ++) {
		hcDist[i] = (float)hc[i] / (heNum * hbSize);
	}
	LOG4CXX_INFO(logger, "sample dist " << hcDist);

	///////////////////////// 2nd half /////////////////////////////////////////////////

	std::cout << "--------------------------------------------------> 2nd half" << std::endl;
	for (int i = capacity / 2; i < capacity; i ++) {
		rb.put({i}, {i * 1.0}, {i}, {i}, {false}, 1);
	}
	std::cout << "Put 2nd half" << std::endl;
	std::cout << "sum = " << rb.getSum() << std::endl;

	torch::Tensor halfPrios2 = torch::rand({capacity / 2});
	std::cout << "update half: " << std::endl << halfPrios << std::endl;

	torch::Tensor wholePrios = torch::zeros({2, capacity / 2});
	wholePrios[0].copy_(halfPrios);
	wholePrios[1].copy_(halfPrios2);
	std::cout << "exp sum = " << wholePrios.sum() << std::endl;
	torch::Tensor wholeDist = wholePrios / wholePrios.sum();
	std::cout << "exp dist = " << wholeDist << std::endl;

	float* hp2 = halfPrios2.data_ptr<float>();
	for (int i = capacity / 2; i < capacity; i ++) {
		rb.updatePrios({i}, {hp2[i - capacity / 2]});
	}
	std::cout << "sum = " << rb.getSum() << std::endl;

	std::vector<int> wc(capacity, 0);
	const int weNum = 100;
	const int wbSize = capacity;
	for (int i = 0; i < weNum; i ++) {
		auto rc = rb.sampleBatch(wbSize);
		auto idx = std::get<0>(rc);
//		LOG4CXX_DEBUG(logger, "sample idx " << idx);
		for (const auto& id: idx) {
			wc[id] ++;
		}
	}
	LOG4CXX_INFO(logger, "sample nums " << wc);
	std::vector<float> wcDist(wc.size(), 0);
	for (int i = 0; i < wc.size(); i ++) {
		wcDist[i] = (float)wc[i] / (weNum * wbSize);
	}
	LOG4CXX_INFO(logger, "sample dist " << wcDist);
}

void testMin() {
	const int capacity = 8;
	PrioRb rb(capacity);

	for (int i = 0; i < capacity / 2; i ++) {
		rb.put({i}, {i * 1.0}, {i}, {i}, {false}, 1);
	}
	std::cout << "Put half" << std::endl;
	std::cout << "half min = " << rb.getMinW() << std::endl;

	torch::Tensor halfPrios = torch::rand({capacity / 2});
	std::cout << "update half: " << std::endl << halfPrios << std::endl;

	float* hp = halfPrios.data_ptr<float>();
	for (int i = 0; i < capacity / 2; i ++) {
		rb.updatePrios({i}, {hp[i]});
	}
	std::cout << "min = " << rb.getMinW() << std::endl;

	/////////////////////////////////////////////// 2nd half
	std::cout << "------------------------------------_> 2nd half" << std::endl;

	std::cout << "put 2nd half" << std::endl;
	for (int i = capacity / 2; i < capacity; i ++) {
		rb.put({i}, {i * 1.0}, {i}, {i}, {false}, 1);
	}

	torch::Tensor halfPrios2 = torch::rand({capacity / 2});
	std::cout << "update half: " << std::endl << halfPrios2 << std::endl;
	float* hp2 = halfPrios2.data_ptr<float>();
	for (int i = capacity / 2; i < capacity; i ++) {
		rb.updatePrios({i}, {hp2[i - capacity / 2]});
	}
	std::cout << "min = " << rb.getMinW() << std::endl;
}

void testUpdate() {
	const int capacity = 8;
	PrioRb rb(capacity);

	for (int i = 0; i < capacity; i ++) {
		rb.put({i}, {i * 1.0}, {i}, {i}, {false}, 1);
	}
	torch::Tensor prios0 = torch::rand({capacity});
	std::cout << "update 0: " << prios0 << std::endl;
	std::vector<int> indices(capacity, 0);
	for (int i = 0; i < capacity; i ++) {
		indices[i] = i;
	}
	std::vector<float> priosVec0(prios0.data_ptr<float>(), prios0.data_ptr<float>() + capacity);
	rb.updatePrios(indices, priosVec0);
	std::cout << "min = " << rb.getMinW() << std::endl;
	std::cout << "sum = " << rb.getSum() << std::endl;
	std::cout << "exp min = " << prios0.min() << std::endl;
	std::cout << "exp sum = " << prios0.sum() << std::endl;

	std::cout << "----------------------------> Update 1" << std::endl;
	torch::Tensor prios1 = torch::rand({capacity});
	torch::Tensor dist1 = torch::rand({capacity});
	std::cout << "update prios: " << prios1 << std::endl;
	for (int i = 0; i < capacity; i ++) {
		int index = dist1.data_ptr<float>()[i] * capacity;
		float prio = prios1.data_ptr<float>()[i];

		rb.updatePrios({index}, {prio});
		std::cout << "min = " << rb.getMinW() << std::endl;
		std::cout << "sum = " << rb.getSum() << std::endl;
	}
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
    log4cxx::LogManager::getLoggerRepository()->setConfigured(true);
}
}

int main(int argc, char** argv) {
	logConfigure(false);

//	testSum();
//	testMin();
	testUpdate();
//	test1(atoi(argv[1]), atoi(argv[2]));
//	test2(atoi(argv[1]), atoi(argv[2]));
//	test3(atoi(argv[1]), atoi(argv[2]));
//	test4(atoi(argv[1]), atoi(argv[2]));
//	test5(atoi(argv[1]), atoi(argv[2]));
//	test6(atoi(argv[1]), atoi(argv[2]));
//	test7(atoi(argv[1]), atoi(argv[2]));
//	test8(atoi(argv[1]), atoi(argv[2]));
//	test9(atoi(argv[1]), atoi(argv[2]));
//	test10(atoi(argv[1]), atoi(argv[2]));
//	test11(atoi(argv[1]), atoi(argv[2]));
//	test12(atoi(argv[1]), atoi(argv[2]));
//	test13(atoi(argv[1]), atoi(argv[2]));
//	test14(atoi(argv[1]), atoi(argv[2]));
//	test15(atoi(argv[1]), atoi(argv[2]));
//	test16(atoi(argv[1]), atoi(argv[2]));
//	test17(atoi(argv[1]), atoi(argv[2]));
//	test18(atoi(argv[1]), atoi(argv[2]));
//	test19(atoi(argv[1]), atoi(argv[2]));
//	test27(atoi(argv[1]), atoi(argv[2]));
//	testCal(atoi(argv[1]), atoi(argv[2]));

//	testtest14(atoi(argv[1]), atoi(argv[2]));

	LOG4CXX_INFO(logger, "End of test");
}
