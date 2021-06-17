/*
 * priorb.h
 *
 *  Created on: May 23, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_UTILS_PRIORB_H_
#define INC_GYMTEST_UTILS_PRIORB_H_

#include <vector>
#include <random>
#include <algorithm>    // std::random_shuffle
#include <vector>
#include <log4cxx/logger.h>

class PrioRb {
private:
	int index = 0;

	int indiceIndex = 0;
	int count = 0;

	std::vector<float> prioSumTree;
	std::vector<int> minHeap;
	std::vector<int> prio2MinHeap;

	float defaultMaxPrio = 1;

	int searchSumTree(float sumPrefix);
	void updateSumTree(const int index, const float prio);

	void updateMinHeap(const int index);
	void updatePrio(const int index, const float prio);

	log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("priorb");

public:
	const int capacity;

	std::vector<std::vector<float>> states;
	std::vector<std::vector<float>> nextStates;
	std::vector<float> rewards;
	std::vector<long> actions;
	std::vector<bool> dones;

	PrioRb(int cap);
	~PrioRb() = default;
	PrioRb(const PrioRb&) = delete;

	//TODO: optimize copy
	void put(
			std::vector<float> state, std::vector<float> nextState,
			std::vector<float> reward, std::vector<long> action, std::vector<bool> done, const int batchSize);
	std::tuple<std::vector<int>, std::vector<float>> sampleBatch(int batchSize);
	inline int getCount() { return count; }

	float getMinW();
	float getSum();

	void updatePrios(const std::vector<int>& indices, const std::vector<float>& prios);
};



#endif /* INC_GYMTEST_UTILS_PRIORB_H_ */
