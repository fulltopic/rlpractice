/*
 * testseqreader.cpp
 *
 *  Created on: Mar 23, 2021
 *      Author: zf
 */


#include <dbtools/lmdbseqreader.h>

#include <string>

#include <log4cxx/basicconfigurator.h>

log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("test"));

int main() {
	log4cxx::BasicConfigurator::configure();

	const std::string dbPath = "/run/media/zf/Newsmy/mjres/pf4/test/testdb";

	LmdbSeqReader reader(dbPath);
//	reader.next();
	reader.next(2);
//	reader.nextWithSeq(3, 2);

	LOG4CXX_INFO(logger, "End test");
}

