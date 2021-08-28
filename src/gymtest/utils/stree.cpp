/*
 * stree.cpp
 *
 *  Created on: Aug 26, 2021
 *      Author: zf
 */


#include "gymtest/utils/stree.h"
#include <cmath>
#include <iostream>
#include <assert.h>

SegTree::SegTree(int iCap): cap(iCap), dataCap(iCap * 2), re(rd()), datas(iCap * 2, 0.0f) {
	assert(cap > 0);

	int tCap = cap;
	while (tCap > 0) {
		assert(((tCap % 2) == 0) || (tCap == 1));
		tCap /= 2;
	}
}

void SegTree::add(float prio) {
//	std::cout << "SegTree add" << std::endl;
	int nextIndex = (curIndex + 1) % cap;

	int curDataIndex = getDataIndex(curIndex);
	float diff =  prio - datas[curDataIndex];
	datas[curDataIndex] = prio;

	curIndex = nextIndex;
	if (len < cap) {
		len ++;
	}

//	std::cout << "add " << curDataIndex << std::endl;
	adjust(curDataIndex, diff);

	//check not happen on first add
	if (curIndex == 0) {
		check();
	}
//	check();
}

void SegTree::update(int index, float prio) {
	assert(index < len);
	int dataIndex = getDataIndex(index);
	float diff = prio - datas[dataIndex];
	datas[dataIndex] = prio;

	adjust(dataIndex, diff);

//	check();
}

void SegTree::update(std::vector<int> indices, std::vector<float> prios) {
	for (int i = 0; i < indices.size(); i ++) {
		update(indices[i], prios[i]);
	}
}

std::pair<std::vector<int>, std::vector<float>> SegTree::sample(int batchSize) {
	assert(batchSize < len);

	float sumPrio = datas[1];

	std::vector<int> indice(batchSize, 0);
	std::vector<float> prios(batchSize, 0);

	for (int i = 0; i < batchSize; i ++) {
		float bPrio = i * sumPrio / batchSize;
		float ePrio = (i + 1) * sumPrio / batchSize;
//		std::cout << "generated segment: " << bPrio << ", " << ePrio << std::endl;

		std::uniform_real_distribution<> dist(bPrio, ePrio);
		float expPrio = (float)dist(re);
//		std::cout << "expPrio = " << expPrio << std::endl;

		int dataIndex = sample(expPrio);

		indice[i] = getIndex(dataIndex);
		prios[i] = datas[dataIndex];
	}

	return {indice, prios};
}

void SegTree::adjust(int dataIndex, float diff) {
//	std::cout << "adjust to " << dataIndex << std::endl;
	int pIndex = dataIndex / 2;

	while (pIndex > 0) {
//		std::cout << "adjust to " << pIndex << std::endl;
//		float origP = datas[pIndex];
//		float origL = datas[pIndex * 2];
//		float origR = datas[pIndex * 2 + 1];

		datas[pIndex] += diff;

//		if (datas[pIndex] != (datas[pIndex * 2] + datas[pIndex * 2 +1])) {
//			std::cout << "Mismatch " << pIndex << ": " << datas[pIndex] << " != " << (datas[pIndex * 2] + datas[pIndex * 2 + 1]) << std::endl;
//			assert(datas[pIndex] == (datas[pIndex * 2] + datas[pIndex * 2 + 1]));
//		}

		pIndex = pIndex / 2;
	}
}

int SegTree::sample(const float expSum) {
	int cIndex = 1;

	float sum = expSum;
	while (cIndex < cap) { //TODO: not datacap, but real len
//		std::cout << "cIndex = " << cIndex << std::endl;
		int lIndex = cIndex * 2;
		int rIndex = cIndex * 2 + 1;

		if (sum <= datas[lIndex]) {
			cIndex = lIndex;
		} else {
			sum -= datas[lIndex];
			cIndex = rIndex;
		}
	}

//	assert(cIndex < (cap + len));
	//some float issue
	if (cIndex >= (cap + len)) {
		return (cap + len - 1);
	}
	return cIndex;
}

//Recalculate to remove accumulated float issue
void SegTree::check() {
//	for (int i = cap - 1; i > 0; i --) {
//		if (datas[i] != (datas[i * 2] + datas[i * 2 + 1])) {
//			std::cout << "Mismatch " << i << ": " << datas[i] << " != " << (datas[i * 2] + datas[i * 2 + 1]) << std::endl;
//			assert(datas[i] == (datas[i * 2] + datas[i * 2 + 1]));
//		}
//	}

	for (int i = cap - 1; i > 0; i --) {
		datas[i] = datas[i * 2] + datas[i * 2 - 1];
	}
}

std::ostream& operator<< (std::ostream& os, const SegTree& st) {
//	os << "SegTree datas: " << std::endl;
	for (int i = 0; i < st.datas.size(); i ++) {
		os << st.datas[i] << ", ";
		if ((i + 1) % 16 == 0) {
			os << std::endl;
		}
	}

	return os;
}
