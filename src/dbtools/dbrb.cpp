/*
 * dbrb.cpp
 *
 *  Created on: Apr 26, 2021
 *      Author: zf
 */



#include "dbtools/dbrb.h"

LmdbRb::LmdbRb(std::string path, int capacity, int sSize): cap(capacity),dbPath(path), stateSize(sSize),
//	dbRead(caffe2::db::CreateDB(dbType, path, caffe2::db::READ)),
//	dbWrite(caffe2::db::CreateDB(dbType, path, caffe2::db::WRITE)),
	uniDist(0, capacity)
{
	dbWrite = caffe2::db::CreateDB(dbType, path, caffe2::db::WRITE);
	if (dbWrite == nullptr) {
		dbWrite = caffe2::db::CreateDB(dbType, path, caffe2::db::NEW);
	}
	dbRead = caffe2::db::CreateDB(dbType, path, caffe2::db::READ);

	std::unique_ptr<caffe2::db::Cursor> cursor(dbRead->NewCursor());
	cursor->SeekToFirst();
	while (cursor->Valid()) {
		count ++;
		cursor->Next();
	}

	count = std::max<int>(count, cap);
	putIndex = (putIndex + 1) % cap;
}

void LmdbRb::put(std::vector<float> state, std::vector<float> nextState,
		std::vector<float> reward, std::vector<long> action, std::vector<bool> done, int batchSize) {
	std::unique_ptr<caffe2::db::Transaction> trans = dbWrite->NewTransaction();

	float* stateP = state.data();
	float* nextStateP = nextState.data();

	for (int i = 0; i < batchSize; i ++) {
		int keyValue = putIndex;
		char* keyChars = reinterpret_cast<char*>(&keyValue);
		std::string key(keyChars, sizeof(int));

		putIndex = (putIndex + 1) % cap;

		char* stateChars = reinterpret_cast<char*>(stateP);
		std::string value(stateChars, sizeof(float) * stateSize);
		stateP += stateSize;
		char* nextStateChars = reinterpret_cast<char*>(nextStateP);
		value += std::string(nextStateChars, sizeof(float) * stateSize);
		nextStateP += stateSize;

		float r = reward[i];
		char* rChars = reinterpret_cast<char*>(&r);
		value += std::string(rChars, sizeof(float));

		int a = action[i];
		char* aChars = reinterpret_cast<char*>(&a);
		value += std::string(aChars, sizeof(int));

		int isDone = done[i]? 1: 0;
		char* doneChars = reinterpret_cast<char*>(&isDone);
		value += std::string(doneChars, sizeof(int));

		trans->Put(key, value);
	}

	trans->Commit();
	//reset by destructor

	count = std::max<int>(count + batchSize, cap);
}

std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>,
				std::vector<float>, std::vector<long>, std::vector<bool>> LmdbRb::getBatch(int batchSize) {
	std::vector<std::vector<float>> states;
	states.reserve(batchSize);
	std::vector<std::vector<float>> nextStates;
	nextStates.reserve(batchSize);
	std::vector<float> rewards(batchSize, 0);
	std::vector<long> actions(batchSize, 0);
	std::vector<bool> dones(batchSize, false);

	std::unique_ptr<caffe2::db::Cursor> cursor(dbRead->NewCursor());
	int num = 0;
	do {
		int index = uniDist(e) % count;
		char* keyIndex = reinterpret_cast<char*>(&index);
		std::string key(keyIndex, sizeof(int));

		cursor->Seek(key);
		if (cursor->Valid()) {
			LOG4CXX_DEBUG(logger, "Found key " << key);

			std::string vStr = cursor->value();
			char* vChars = vStr.data();

			float* stateP = reinterpret_cast<float*>(vChars);
			std::vector<float> state(stateP, stateP + stateSize);
			std::vector<float> nextState(stateP + stateSize, stateP + stateSize * 2);
			float reward = *(stateP + stateSize * 2);

			vChars += sizeof(float) * (stateSize * 2 + 1);
			int* actionP = reinterpret_cast<int*>(vChars);
			int action = *actionP;

			int* doneP = actionP + 1;
			bool done = (*doneP > 0.5)? true: false;

			states.push_back(state);
			nextStates.push_back(nextState);
			rewards[num] = reward;
			actions[num] = action;
			dones[num] = done;

			num ++;
		} else {
			LOG4CXX_DEBUG(logger, "Failed to found record of " << index);
		}
	} while (num < batchSize);

	return {states, nextStates, rewards, actions, dones};
}

