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
	struct HeapNode {
		ValueType value;
		int defIndex;
	};

	MHeap(int iCap, FuncType func);
	~MHeap() = default;
	MHeap(const MHeap&) = delete;

	void add(const ValueType& value);
	void update(int index, const ValueType& value);

	ValueType getM();
	ValueType replaceTop(const ValueType& minValue);
private:
	const int cap;
	FuncType pred;
//	std::vector<ValueType> datas;
//	std::vector<int> indices;
	std::vector<HeapNode> datas;
	std::vector<int> dataIndices;

//	int len = 0;
	int curIndex = 0;

	void adjust(const int dataIndex);

public:
		friend std::ostream& operator<< (std::ostream& os, const HeapNode& node) {
			os << node.defIndex << "-" << node.value;
			return os;
		}

		friend std::ostream& operator<< (std::ostream& os, const MHeap<ValueType, FuncType>& heap) {
			for (int i = 0; i < heap.datas.size(); i ++) {
				os << heap.datas[i] << ", ";
				if ((i + 1) % 16 == 0) {
					os << std::endl;
				}
			}
			std::cout << std::endl;

			for (int i = 0; i < heap.datas.size(); i ++) {
				os << heap.dataIndices[i] << ", ";
				if ((i + 1) % 16 == 0) {
					os << std::endl;
				}
			}

			return os;
		}
};


template <typename ValueType, typename FuncType>
MHeap<ValueType, FuncType>::MHeap(int iCap, FuncType func): cap(iCap), pred(func) {
	datas.reserve(cap);
	dataIndices.reserve(cap);
}

template <typename ValueType, typename FuncType>
void MHeap<ValueType, FuncType>::add(const ValueType& value) {
	int adjustIndex = curIndex;

	if (datas.size() < cap) {
//		datas.push_back(value);
//		indices.push_back(datas.size() - 1);

		datas.push_back({value, curIndex});
		dataIndices.push_back(curIndex);

//		std::cout << "heap push curIndex: " << datas.size() - 1 << std::endl;
	} else {
//		datas[curIndex] = value;

		int dataIndex = dataIndices[curIndex];
		datas[dataIndex] = {value, curIndex};
		adjustIndex = dataIndex;

//		std::cout << "heap add curIndex: " << curIndex << std::endl;
	}


//	adjust(curIndex);
	adjust(adjustIndex);

	int nextIndex = (curIndex + 1) % cap;
	curIndex = nextIndex;
}

//TODO: try ValueType&&
template <typename ValueType, typename FuncType>
void MHeap<ValueType, FuncType>::update(int index, const ValueType& value) {
	//TODO: assert index
//	datas[index] = value; //copy
//	adjust(index);

	int dataIndex = dataIndices[index];
	datas[dataIndex] = {value, index};

	adjust(dataIndex);
}

template <typename ValueType, typename FuncType>
ValueType MHeap<ValueType, FuncType>::getM() {
	//TODO: assert size
	if (datas.size() < 1) {
		return 0;
	}
	return datas[0].value;
}

template <typename ValueType, typename FuncType>
ValueType MHeap<ValueType, FuncType>::replaceTop(const ValueType& minValue) {
	int defIndex = datas[0].defIndex;
	auto value = datas[0].value;

	datas[0] = {minValue, defIndex};
	adjust(0);

	return value;
}

template <typename ValueType, typename FuncType>
void MHeap<ValueType, FuncType>::adjust(const int dataIndex) {
//	std::cout << "adjust " << index << " = " << datas[index] << std::endl;
	if (datas.size() <= 1) {
		return;
	}
//	assert(dataIndex < datas.size());

	//TODO: update data by indices[index]
//	int cIndex = index;
	int cIndex = dataIndex;
	while (cIndex > 0) {
//		std::cout << "Try to upward" << std::endl;
		int pIndex = (cIndex - 1) / 2;
		if (pred(datas[cIndex].value, datas[pIndex].value)) {
			std::swap(datas[cIndex], datas[pIndex]);
			std::swap(dataIndices[datas[cIndex].defIndex], dataIndices[datas[pIndex].defIndex]);
			cIndex = pIndex;
//			std::cout << "upward to " << cIndex << std::endl;
		} else {
			break;
		}
	}
	if (cIndex != dataIndex) {
//		std::cout << "updated upward" << std::endl;
		return;
	}

	cIndex = dataIndex; //not necessary
	while (cIndex < datas.size()) {
		int lIndex = cIndex * 2 + 1;
		if (lIndex >= datas.size()) {
			break;
		}
		int nIndex = cIndex;
		if (pred(datas[lIndex].value, datas[cIndex].value)) {
			nIndex = lIndex;
		}

		int rIndex = cIndex * 2 + 2;
		if (rIndex < datas.size()) {
			if (pred(datas[rIndex].value, datas[nIndex].value)) {
				nIndex = rIndex;
			}
		}

		if (nIndex != cIndex) {
			std::swap(datas[nIndex], datas[cIndex]);
			std::swap(dataIndices[datas[nIndex].defIndex], dataIndices[datas[cIndex].defIndex]);
			cIndex = nIndex;
		} else {
			break;
		}
	}
}
#endif /* INC_GYMTEST_UTILS_MHEAP_HPP_ */
