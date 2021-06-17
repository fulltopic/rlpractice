/*
 * testdbrb.cpp
 *
 *  Created on: Apr 26, 2021
 *      Author: zf
 */


#include "dbtools/dbrb.h"

#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/simplelayout.h>
#include <log4cxx/logmanager.h>

namespace {
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("lmdbtest"));

void test0() {
	const int stateSize = 16;
	const int cap = 16;
	const std::string dbPath = "./test0";
	LmdbRb rb(dbPath, stateSize, cap);

	const int batchSize = 4;
	for (int i = 0; i < batchSize; i ++) {
		std::vector<float> state(stateSize, i);
		std::vector<float> nextState(stateSize, i * 2);
		std::vector<float> reward(1, i * 3);
		std::vector<long> action(1, i);
		std::vector<bool> done(1, false);

		rb.put(state, nextState, reward, action, done, 1);
	}
}

void test1() {
	const int stateSize = 16;
	const int cap = 16;
	const std::string dbPath = "./test0";
	LmdbRb rb(dbPath, stateSize, cap);

	LOG4CXX_INFO(logger, "record num: " << rb.getCount());
	int batchSize = 2;
	auto rc = rb.getBatch(batchSize);
	for (int i = 0; i < batchSize; i ++) {
		std::vector<float> state = std::get<0>(rc)[i];
		std::vector<float> nextState = std::get<1>(rc)[i];
		float reward = std::get<2>(rc)[i];
		long action = std::get<3>(rc)[i];
		bool done = std::get<4>(rc)[i];

		LOG4CXX_INFO(logger, "get state: " << state);
		LOG4CXX_INFO(logger, "get next state: " << nextState);
		LOG4CXX_INFO(logger, "others: " << reward << ", " << action << ", " << done);
	}
}

//TODO: put batch > 1
void test2() {
	const int stateSize = 16;
	const int cap = 1024;
	const int epicNum = 128;
	const int batchSize = 4;
	const std::string dbPath = "./test0";
	LmdbRb rb(dbPath, stateSize, cap);

	std::random_device r = std::random_device();
	std::default_random_engine e = std::default_random_engine(r());
	std::uniform_real_distribution<float> uniDist(0, 1);

	for (int i = 0; i < epicNum; i ++) {
		float reward = uniDist(e);
		int action = i;
		bool done = (action % 2) == 0? true: false;
		std::vector<float> state(stateSize, 0);
		std::vector<float> nextState(stateSize, 0);
		for (int j = 0; j < stateSize; j ++) {
			state[j] = reward * 2;
			nextState[j] = reward * 3;
		}

		rb.put(state, nextState, {reward}, {action}, {done}, 1);

		if (rb.getCount() >= batchSize) {
			auto rc = rb.getBatch(batchSize);
			for (int j = 0; j < batchSize; j ++) {
				std::vector<float> s = std::get<0>(rc)[j];
				std::vector<float> ns = std::get<1>(rc)[j];
				float r = std::get<2>(rc)[j];
				long a = std::get<3>(rc)[j];
				bool d = std::get<4>(rc)[j];

				LOG4CXX_INFO(logger, "check action: " << a);
				for (int k = 0; k < stateSize; k ++) {
					if (s[k] != r * 2) {
						LOG4CXX_ERROR(logger, "state not match: " << s[k] << " != " << (reward * 2));
					}
					if (ns[k] != r * 3) {
						LOG4CXX_ERROR(logger, "nextState not match: " << s[k] << " != " << (reward * 3));
					}
				}
				bool ed = (a % 2) == 0? true: false;
				if (d != ed) {
					LOG4CXX_ERROR(logger, "done not match: " << d << " != " << ed);
				}
			}
		}
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
    //	log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getDebug());
    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getInfo());
    log4cxx::LogManager::getLoggerRepository()->setConfigured(true);
}
}

int main() {
	logConfigure(false);

//	test0();
//	test1();
	test2();
}
