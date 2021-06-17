/*
 * nobatchrb.h
 *
 *  Created on: Apr 11, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_UTILS_NOBATCHRB_H_
#define INC_GYMTEST_UTILS_NOBATCHRB_H_

#include <vector>
#include <random>
#include <algorithm>    // std::random_shuffle
#include <vector>

class NoBatchRB {
private:
	int index = 0;
//	std::uniform_int_distribution<int> distribution;
//	std::default_random_engine generator;

	std::vector<int> indices;
	int indiceIndex = 0;
	int count = 0;

public:
	const int capacity;

	std::vector<std::vector<float>> states;
	std::vector<std::vector<float>> nextStates;
	std::vector<float> rewards;
	std::vector<long> actions;
	std::vector<bool> dones;

	NoBatchRB(int cap);
	~NoBatchRB() = default;

	//TODO: optimize copy
	void put(
			std::vector<float> state, std::vector<float> nextState, std::vector<float> reward, std::vector<long> action, std::vector<bool> done, const int batchSize);
	int randSelect();
	inline int getCount() { return count; }
};



#endif /* INC_GYMTEST_UTILS_NOBATCHRB_H_ */
