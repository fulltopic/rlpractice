/*
 * replaybuffer.h
 *
 *  Created on: Apr 8, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_UTILS_REPLAYBUFFER_H_
#define INC_GYMTEST_UTILS_REPLAYBUFFER_H_

#include <vector>
#include <random>
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
//#include <ctime>        // std::time
//#include <cstdlib>

class ReplayBuffer {
private:
	int index;
	std::uniform_int_distribution<int> distribution;
	std::default_random_engine generator;

	std::vector<int> indices;
	int indiceIndex;

	int count = 0;
public:
	const int capacity;

	std::vector<std::vector<float>> states;
	std::vector<std::vector<float>> nextStates;
	std::vector<std::vector<float>> rewards;
	std::vector<std::vector<long>> actions;
	std::vector<std::vector<bool>> dones;

	ReplayBuffer(int cap);
	~ReplayBuffer() = default;

	//TODO: optimize copy
	void put(std::vector<float> state, std::vector<float> nextState, std::vector<float> reward, std::vector<long> action, std::vector<bool> done);
	int randSelect();
	int getIndex();

	inline int getCount() { return count; }
};



#endif /* INC_GYMTEST_UTILS_REPLAYBUFFER_H_ */
