/*
 * testdbwrite.cpp
 *
 *  Created on: Apr 25, 2021
 *      Author: zf
 */


#include <string>
#include <vector>
#include <memory>
#include <limits>

//#include <torch/torch.h>

#include <caffe2/core/db.h>
#include <caffe2/core/init.h>

#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/simplelayout.h>
#include <log4cxx/logmanager.h>

namespace {
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("lmdbtest"));

const std::string dbType = "lmdb";
//write simple
void test0() {
	std::string dbPath = "./db0";

	std::unique_ptr<caffe2::db::DB> db = caffe2::db::CreateDB(dbType, dbPath, caffe2::db::WRITE);
//	LOG4CXX_INFO(logger, "null db? " << (db == nullptr));
//	std::unique_ptr<caffe2::db::Cursor> cursor;
//	cursor = db->NewCursor();
//	cursor->SeekToFirst();
	std::unique_ptr<caffe2::db::Transaction> trans(nullptr);
	LOG4CXX_INFO(logger, "Try to create null trans");
	trans = db->NewTransaction();
	LOG4CXX_INFO(logger, "Try to get a transferred trans");
//
	std::string key = "1";
	std::string value = "test1";
//
	trans->Put(key, value);
	LOG4CXX_INFO(logger, "Try to put something");
	trans->Commit();
	LOG4CXX_INFO(logger, "Try to commit");

	trans.reset();
	db.reset();
//	db->Close();
}

//read simple
void test1() {
	std::string dbPath = "./db0";

	std::unique_ptr<caffe2::db::DB> db = caffe2::db::CreateDB(dbType, dbPath, caffe2::db::READ);
//	LOG4CXX_INFO(logger, "null db? " << (db == nullptr));
	std::unique_ptr<caffe2::db::Cursor> cursor;
	cursor = db->NewCursor();
	cursor->SeekToFirst();
//	std::unique_ptr<caffe2::db::Transaction> trans(nullptr);
//	LOG4CXX_INFO(logger, "Try to create null trans");
//	trans = db->NewTransaction();
//	LOG4CXX_INFO(logger, "Try to get a transferred trans");
//
	std::string key = "1";
	std::string value = "test1";
//
	LOG4CXX_INFO(logger, "Get first: " << cursor->key() << ", " << cursor->value());
	db.reset();
//	db->Close();
}

//string from chars
void test2() {
	char cs[] {'a', 'b', 'c'};
	std::string s(cs);
	LOG4CXX_INFO(logger, "s: " << s);

	int keyValue = 36;
	char* keyChars = reinterpret_cast<char*>(&keyValue);
	std::string ks(keyChars);
	LOG4CXX_INFO(logger, "key: " << ks);
}

//write vector
void test3() {
	std::string dbPath = "./db0";

	std::unique_ptr<caffe2::db::DB> db = caffe2::db::CreateDB(dbType, dbPath, caffe2::db::WRITE);
//	LOG4CXX_INFO(logger, "null db? " << (db == nullptr));
//	std::unique_ptr<caffe2::db::Cursor> cursor;
//	cursor = db->NewCursor();
//	cursor->SeekToFirst();
	std::unique_ptr<caffe2::db::Transaction> trans = db->NewTransaction();
	LOG4CXX_INFO(logger, "Try to get a transferred trans");
//
	int keyValue = 7;
	char* keyChars = reinterpret_cast<char*>(&keyValue);
	std::string key(keyChars, sizeof(int));

	std::vector<float> vValue(4, 6);
	for (int i = 0; i < vValue.size(); i ++) {
		vValue[i] = i + 7;
	}
	char* vChars = reinterpret_cast<char*>(vValue.data());
	std::string value(vChars, sizeof(float) * vValue.size());
//
	trans->Put(key, value);
	LOG4CXX_INFO(logger, "Try to put something");
	trans->Commit();
	LOG4CXX_INFO(logger, "Try to commit");

	trans.reset();
	db.reset();
//	db->Close();
}

//read db
void test4() {
	std::string dbPath = "./db0";

	std::unique_ptr<caffe2::db::DB> db = caffe2::db::CreateDB(dbType, dbPath, caffe2::db::READ);
//	LOG4CXX_INFO(logger, "null db? " << (db == nullptr));
	std::unique_ptr<caffe2::db::Cursor> cursor;
	cursor = db->NewCursor();
	cursor->SeekToFirst();

	while (cursor->Valid()) {
		std::string ks = cursor->key();
		if (ks.compare("1") == 0) {
			std::string vs = cursor->value();
			LOG4CXX_INFO(logger, "read: " << ks << "-->" << vs);
		} else {
			char* kc = ks.data();
			int* kv = reinterpret_cast<int*>(kc);

			std::string vs = cursor->value();
			char* vc = vs.data();
			float* vf = reinterpret_cast<float*>(vc);
			std::vector<float> value(vf, vf + 4);

			LOG4CXX_INFO(logger, "read: " << *kv << "-->" << value);
		}

		cursor->Next();
	}

	db.reset();
//	db->Close();
}

void test5() {
	std::string dbPath = "./db0";

	std::unique_ptr<caffe2::db::DB> db = caffe2::db::CreateDB(dbType, dbPath, caffe2::db::WRITE);
//	LOG4CXX_INFO(logger, "null db? " << (db == nullptr));
//	std::unique_ptr<caffe2::db::Cursor> cursor;
//	cursor = db->NewCursor();
//	cursor->SeekToFirst();
	std::unique_ptr<caffe2::db::Transaction> trans = db->NewTransaction();
	LOG4CXX_INFO(logger, "Try to get a transferred trans");
//
	int keyValue = 7;
	char* keyChars = reinterpret_cast<char*>(&keyValue);
	std::string key(keyChars, sizeof(int));

	std::vector<float> vValue(4, 6);
	for (int i = 0; i < vValue.size(); i ++) {
		vValue[i] = i + 7;
	}
	char* vChars = reinterpret_cast<char*>(vValue.data());
	std::string value(vChars, sizeof(float) * vValue.size());

	int action = 17;
	char* aChars = reinterpret_cast<char*>(&action);
	value += std::string(aChars, sizeof(int));

	bool done = true;
	char doneC = (done)? 1: 0;
	value += std::string(&doneC, sizeof(char));

	std::vector<float> nextValue(4, 12685);
	char* nextChars = reinterpret_cast<char*>(nextValue.data());
	value += std::string(nextChars, sizeof(float) * nextValue.size());

//
	trans->Put(key, value);
	LOG4CXX_INFO(logger, "Try to put something");
	trans->Commit();
	LOG4CXX_INFO(logger, "Try to commit");

	trans.reset();
	db.reset();
//	db->Close();
}

void test6() {
	std::string dbPath = "./db0";

	std::unique_ptr<caffe2::db::DB> db = caffe2::db::CreateDB(dbType, dbPath, caffe2::db::READ);
//	LOG4CXX_INFO(logger, "null db? " << (db == nullptr));
	std::unique_ptr<caffe2::db::Cursor> cursor;
	cursor = db->NewCursor();
	cursor->SeekToFirst();

	int kValue = 7;
	char* kc = reinterpret_cast<char*>(&kValue);
	cursor->Seek(std::string(kc, sizeof(int)));
	if (cursor->Valid()) {
		LOG4CXX_INFO(logger, "Found key " << kValue);

		std::string vs = cursor->value();
		char* vc = vs.data();
		float* vf = reinterpret_cast<float*>(vc);
		std::vector<float> value(vf, vf + 4);
		LOG4CXX_INFO(logger, "read value " << value);

		vc = reinterpret_cast<char*>(vf + 4);
		int* actionP = reinterpret_cast<int*>(vc);
		LOG4CXX_INFO(logger, "read action: " << *actionP);

		vc = reinterpret_cast<char*>(actionP + 1);
		char* doneP = vc;
		bool done = *doneP > 0? true: false;
		LOG4CXX_INFO(logger, "is done: " << done);

		vc = reinterpret_cast<char*>(doneP + 1);
		float* nextP = reinterpret_cast<float*>(vc);
		std::vector<float> nextValue(nextP, nextP + 4);
		LOG4CXX_INFO(logger, "next state: " << nextValue);
	} else {
		LOG4CXX_ERROR(logger, "No record found of key " << kValue);
	}


	db.reset();
//	db->Close();
}

void test7() {
	std::string dbPath = "./db0";

	std::unique_ptr<caffe2::db::DB> dbRead = caffe2::db::CreateDB(dbType, dbPath, caffe2::db::READ);
	std::unique_ptr<caffe2::db::DB> dbWrite = caffe2::db::CreateDB(dbType, dbPath, caffe2::db::WRITE);


	std::unique_ptr<caffe2::db::Transaction> trans = dbWrite->NewTransaction();
	LOG4CXX_INFO(logger, "Try to get a transferred trans");
//
	int keyValue = 8;
	char* keyChars = reinterpret_cast<char*>(&keyValue);
	std::string key(keyChars, sizeof(int));

	std::vector<float> vValue(4, 6);
	for (int i = 0; i < vValue.size(); i ++) {
		vValue[i] = i + 8;
	}
	char* vChars = reinterpret_cast<char*>(vValue.data());
	std::string value(vChars, sizeof(float) * vValue.size());

	int action = 18;
	char* aChars = reinterpret_cast<char*>(&action);
	value += std::string(aChars, sizeof(int));

	bool done = false;
	char doneC = (done)? 1: 0;
	value += std::string(&doneC, sizeof(char));

	std::vector<float> nextValue(4, 1288);
	char* nextChars = reinterpret_cast<char*>(nextValue.data());
	value += std::string(nextChars, sizeof(float) * nextValue.size());

//
	trans->Put(key, value);
	LOG4CXX_INFO(logger, "Try to put something");
	trans->Commit();
	LOG4CXX_INFO(logger, "Try to commit");
	trans.reset();


//	LOG4CXX_INFO(logger, "null db? " << (db == nullptr));
	std::unique_ptr<caffe2::db::Cursor> cursor;
	cursor = dbRead->NewCursor();
	cursor->SeekToFirst();

	int kValue = 8;
	char* kc = reinterpret_cast<char*>(&kValue);
	cursor->Seek(std::string(kc, sizeof(int)));
	if (cursor->Valid()) {
		LOG4CXX_INFO(logger, "Found key " << kValue);

		std::string vs = cursor->value();
		char* vc = vs.data();
		float* vf = reinterpret_cast<float*>(vc);
		std::vector<float> value(vf, vf + 4);
		LOG4CXX_INFO(logger, "read value " << value);

		vc = reinterpret_cast<char*>(vf + 4);
		int* actionP = reinterpret_cast<int*>(vc);
		LOG4CXX_INFO(logger, "read action: " << *actionP);

		vc = reinterpret_cast<char*>(actionP + 1);
		char* doneP = vc;
		bool done = *doneP > 0? true: false;
		LOG4CXX_INFO(logger, "is done: " << done);

		vc = reinterpret_cast<char*>(doneP + 1);
		float* nextP = reinterpret_cast<float*>(vc);
		std::vector<float> nextValue(nextP, nextP + 4);
		LOG4CXX_INFO(logger, "next state: " << nextValue);
	} else {
		LOG4CXX_ERROR(logger, "No record found of key " << kValue);
	}


//	db.reset();
//	db->Close();
}
void testLimit() {
	LOG4CXX_INFO(logger, "int limit: " << std::numeric_limits<int>::min() << ", " << std::numeric_limits<int>::max());
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
//	test2();
//	test3();
//	test4();
//	test5();
//	test6();
	test7();

//	testLimit();
}
