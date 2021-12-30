/*
 * stree.h
 *
 *  Created on: Aug 26, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_UTILS_STREE_H_
#define INC_GYMTEST_UTILS_STREE_H_


#include <vector>
#include <random>
#include <iostream>

class SegTree {
public:
	SegTree(int iCap);
	~SegTree() = default;
	SegTree(const SegTree&) = delete;

	void add(float prio);
	void update(int index, float prio);
	void update(std::vector<int> indices, std::vector<float> prios);
	std::pair<std::vector<int>, std::vector<float>> sample(int batchSize);
	int sample(const float expPrio);

	inline int size() { return len; }
	inline float getSum() { return datas[1]; }

	friend std::ostream& operator<< (std::ostream& os, const SegTree& st);

	void check();
private:
	const int cap;
	const int dataCap;
	int curIndex = 0;
	int len = 0;

	std::random_device rd;
	std::mt19937 re;

	std::vector<float> datas;

	void adjust(int dataIndex, float diff);

	inline int getDataIndex(int index) { return index + cap; }
	inline int getIndex(int dataIndex) { return dataIndex - cap; }

	inline int dataSize() { return len + cap;}
};

std::ostream& operator<< (std::ostream& os, const SegTree& st);

#endif /* INC_GYMTEST_UTILS_STREE_H_ */
