/*
 * dbrb.h
 *
 *  Created on: Apr 26, 2021
 *      Author: zf
 */

#ifndef INC_DBTOOLS_DBRB_H_
#define INC_DBTOOLS_DBRB_H_
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <cmath>

#include <log4cxx/logger.h>
#include <caffe2/core/db.h>
#include <caffe2/core/init.h>

class LmdbRb {
private:
	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("lmdbrb");
	std::string dbPath;
	const int cap;
	const int stateSize;
	int count = 0;
	int putIndex = 0;

	std::unique_ptr<caffe2::db::DB> dbRead;
	std::unique_ptr<caffe2::db::DB> dbWrite;
//	std::unique_ptr<caffe2::db::Cursor> cursor;

	std::random_device r = std::random_device();
	std::default_random_engine e = std::default_random_engine(r());
	std::uniform_int_distribution<int> uniDist;
public:
	const std::string dbType = "lmdb";
	explicit LmdbRb(std::string path, int capacity, int sSize);
	~LmdbRb() = default;
	LmdbRb(const LmdbRb& ) = delete;

	void put(std::vector<float> state, std::vector<float> nextState,
			std::vector<float> reward, std::vector<long> action, std::vector<bool> done, int batchSize);

	std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>,
					std::vector<float>, std::vector<long>, std::vector<bool>> getBatch(int batchSize);

	inline int getCount() { return count;}
};


#endif /* INC_DBTOOLS_DBRB_H_ */
