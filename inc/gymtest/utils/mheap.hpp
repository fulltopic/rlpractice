/*
 * mheap.hpp
 *
 *  Created on: Aug 25, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_UTILS_MHEAP_HPP_
#define INC_GYMTEST_UTILS_MHEAP_HPP_

#include <vector>
#include <algorithm>
#include <iostream>
#include <string>

template <typename ValueType, typename FuncType>
class MHeap {
public:
	MHeap(int iCap, FuncType func);
	~MHeap() = default;
	MHeap(const MHeap&) = delete;

	void add(const ValueType& value);
	void update(int index, const ValueType& value);

	ValueType getM();
private:
	const int cap;
	FuncType pred;
	std::vector<ValueType> datas;

//	int len = 0;
	int curIndex = 0;

	void adjust(const int index);
};

template <typename ValueType, typename FuncType>
MHeap<ValueType, FuncType>::MHeap(int iCap, FuncType func): cap(iCap), pred(func) {
	datas.reserve(cap);
}

template <typename ValueType, typename FuncType>
void MHeap<ValueType, FuncType>::add(const ValueType& value) {
	if (datas.size() < cap) {
		datas.push_back(value);
	} else {
		datas[curIndex] = value;
	}


	adjust(curIndex);

	int nextIndex = (curIndex + 1) % cap;
	curIndex = nextIndex;
}

template <typename ValueType, typename FuncType>
void MHeap<ValueType, FuncType>::update(int index, const ValueType& value) {
	//TODO: assert index
	datas[index] = value; //copy
	adjust(index);
}

template <typename ValueType, typename FuncType>
ValueType MHeap<ValueType, FuncType>::getM() {
	//TODO: assert size
	if (datas.size() < 1) {
		return 0;
	}
	return datas[0];
}

template <typename ValueType, typename FuncType>
void MHeap<ValueType, FuncType>::adjust(const int index) {
//	std::cout << "adjust " << index << " = " << datas[index] << std::endl;
	//TODO: assert index
	if (datas.size() <= 1) {
		return;
	}

	int cIndex = index;
	while (cIndex > 0) {
//		std::cout << "Try to upward" << std::endl;
		int pIndex = (cIndex - 1) / 2;
		if (pred(datas[cIndex], datas[pIndex])) {
			std::swap(datas[cIndex], datas[pIndex]);
			cIndex = pIndex;
//			std::cout << "upward to " << cIndex << std::endl;
		} else {
			break;
		}
	}
	if (cIndex != index) {
//		std::cout << "updated upward" << std::endl;
		return;
	}

	cIndex = index;
	while (cIndex < datas.size()) {
		int lIndex = cIndex * 2 + 1;
		if (lIndex >= datas.size()) {
			break;
		}
		int nIndex = cIndex;
		if (pred(datas[lIndex], datas[cIndex])) {
			nIndex = lIndex;
		}

		int rIndex = cIndex * 2 + 2;
		if (rIndex < datas.size()) {
			if (pred(datas[rIndex], datas[nIndex])) {
				nIndex = rIndex;
			}
		}

		if (nIndex != cIndex) {
			std::swap(datas[nIndex], datas[cIndex]);
			cIndex = nIndex;
		} else {
			break;
		}
	}
}
#endif /* INC_GYMTEST_UTILS_MHEAP_HPP_ */
