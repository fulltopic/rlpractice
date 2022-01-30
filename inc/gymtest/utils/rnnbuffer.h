/*
 * rnnbuffer.h
 *
 *  Created on: Jan 28, 2022
 *      Author: zf
 */

#ifndef INC_GYMTEST_UTILS_RNNBUFFER_H_
#define INC_GYMTEST_UTILS_RNNBUFFER_H_


#include <vector>
#include <cstdint>
#include <queue>

struct Episode {
	std::vector<std::vector<float>> states;
	std::vector<int64_t> actions;
	std::vector<float> rewards;
	std::vector<float> returns;
};

//class Episodes {
//private:
//	std::queue<Episode> datas;
//
//public:
//	Episodes(int capacity);
//	~Episodes() = default;
//
//
//};


#endif /* INC_GYMTEST_UTILS_RNNBUFFER_H_ */
