/*
 * testmheap.cpp
 *
 *  Created on: Aug 25, 2021
 *      Author: zf
 */


#include "alg/utils/mheap.hpp"


#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include <vector>

namespace {
const int IntRange = 100;

class IntMinPredFunc {
public:
	bool operator()(const int a, const int b)  {
		return a < b;
	}
};

class IntMaxPredFunc {
public:
	bool operator()(const int a, const int b)  {
		return a > b;
	}
};

std::vector<int> genInts(const int count) {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0,IntRange);

  std::vector<int> datas;
  for (int i = 0; i < count; i ++) {
	  int data = distribution(generator);
	  datas.push_back(data);
  }

  return datas;
}

template<typename T>
void printV(const std::string cmt, const std::vector<T>& data) {
	std::cout << cmt << ": " << std::endl;
	for (int i = 0; i < data.size(); i ++) {
		std::cout << data[i] << ", ";
	}
	std::cout << std::endl;
}

void test0() {
	const int num = 20; //TODO: comp time
	const int cap = 10;
	IntMinPredFunc func;
	MHeap<int, IntMinPredFunc> heap(cap, func);

	std::vector<int> datas = genInts(num);
	printV("generated datas", datas);

	for (int i = 0; i < num; i ++) {
		heap.add(datas[i]);
		std::cout << "add " << datas[i] << std::endl;
		std::cout << "min: " << heap.getM() << std::endl;
	}
}

void test1() {
	const int num = 20; //TODO: comp time
	const int cap = 10;
	IntMaxPredFunc func;
	MHeap<int, IntMaxPredFunc> heap(cap, func);

	std::vector<int> datas = genInts(num);
	printV("generated datas", datas);

	for (int i = 0; i < num; i ++) {
		heap.add(datas[i]);
		std::cout << "add " << datas[i] << std::endl;
		std::cout << "max: " << heap.getM() << std::endl;
	}
}

void test2() {
	const int num = 20; //TODO: comp time
	const int cap = 20;
	IntMaxPredFunc func;
	MHeap<int, IntMaxPredFunc> heap(20, func);

	std::vector<int> datas = genInts(num);
	printV("generated datas", datas);

	for (int i = 0; i < num; i ++) {
		heap.add(datas[i]);
//		std::cout << "add " << datas[i] << std::endl;
//		std::cout << "max: " << heap.getM() << std::endl;
	}

	std::cout << "--------------------> sorted " << std::endl;
	for (int i = 0; i < num; i ++) {
		std::cout << heap.getM() << ", " << std::endl;
		heap.update(0, -1);
	}
}
}

int main() {
	test2();
}
