/*
 * testsegtree.cpp
 *
 *  Created on: Aug 26, 2021
 *      Author: zf
 */



#include "gymtest/utils/stree.h"
#include <iostream>
#include <string>
#include <utility>

namespace {
template<typename T>
void printV(std::string cmt, const std::vector<T>& datas) {
	std::cout << cmt << std::endl;
	for (const auto& data: datas) {
		std::cout << data << "," << std::endl;
	}
	std::cout << std::endl;
}

void test0() {
	SegTree st(8);

	st.add(0.5);
	auto rc = st.sample(1);
	auto indice = std::get<0>(rc);
	auto prios = std::get<1>(rc);

	printV("indice", indice);
	printV("prios", prios);
	std::cout << st << std::endl;
}

void test1() {
	SegTree st(8);

	for (int i = 0; i < 8; i ++) {
		st.add(0.5);
		auto rc = st.sample(1);
		auto indice = std::get<0>(rc);
		auto prios = std::get<1>(rc);

		printV("indice", indice);
		printV("prios", prios);
		std::cout << st << std::endl;
	}
}

void test2() {
	SegTree st(4);

	for (int i = 0; i < 4; i ++) {
		st.add(0.5);
		auto rc = st.sample(1);
		auto indice = std::get<0>(rc);
		auto prios = std::get<1>(rc);

		printV("indice", indice);
		printV("prios", prios);
		std::cout << st << std::endl;
	}
	std::cout << "After full --------------------------------" << std::endl;
	for (int i = 0; i < 4; i ++) {
		st.add(0.6);
		auto rc = st.sample(1);
		auto indice = std::get<0>(rc);
		auto prios = std::get<1>(rc);

		printV("indice", indice);
		printV("prios", prios);
		std::cout << st << std::endl;
	}
}

void test3() {
	std::random_device rd;
	std::mt19937 re(rd());
	std::uniform_real_distribution<> dist(0, 10.0);
	SegTree st(8);

	for (int i = 0; i < 8; i ++) {
		float prio = (float)dist(re);
		std::cout << "------> generated prio = " << prio << std::endl;

		st.add(prio);
		auto rc = st.sample(1);
		auto indice = std::get<0>(rc);
		auto prios = std::get<1>(rc);

		printV("indice", indice);
		printV("prios", prios);
		std::cout << st << std::endl;
	}
	std::cout << "After full --------------------------------" << std::endl;
	for (int i = 0; i < 8; i ++) {
		float prio = (float)dist(re);
		std::cout << "------> updated prio = " << prio << std::endl;

		st.update(i, prio);
		auto rc = st.sample(1);
		auto indice = std::get<0>(rc);
		auto prios = std::get<1>(rc);

		printV("indice", indice);
		printV("prios", prios);
		std::cout << st << std::endl;
	}
}

void test4() {
	std::random_device rd;
	std::mt19937 re(rd());
	std::uniform_real_distribution<> dist(0, 10.0);
	SegTree st(8);
	const int num = 8;
	std::vector<float> inputs(num, 0);
	float sum = 0;
	for(int i = 0; i < num; i ++) {
		float prio = (float)dist(re);
		inputs[i] = prio;
		sum += prio;
	}
	for (int i = 0; i < num; i ++) {
		inputs[i] = inputs[i] / sum;
	}

	for (int i = 0; i < num; i ++) {
		st.add(inputs[i]);
		auto rc = st.sample(1);
		auto indice = std::get<0>(rc);
		auto prios = std::get<1>(rc);

		printV("indice", indice);
		printV("prios", prios);
		std::cout << st << std::endl;
	}

	std::cout << "To sample --------------------------------" << std::endl;
	std::vector<int> counts(num, 0);

	for (int i = 0; i < 1000; i ++) {
		auto rc = st.sample(1);
		auto indice = std::get<0>(rc);
		for (const auto& index: indice) {
			counts[index] ++;
		}
	}

	printV("input", inputs);
	printV("counts", counts);
}

void test5() {
	std::random_device rd;
	std::mt19937 re(rd());
	std::uniform_real_distribution<> dist(0, 10.0);
	SegTree st(8);
	const int num = 8;
	float prio = 0.1;

	for (int i = 0; i < num; i ++) {
		st.add(prio);
		prio *= 2;

//		auto rc = st.sample(1);
//		auto indice = std::get<0>(rc);
//		auto prios = std::get<1>(rc);
//
//		printV("indice", indice);
//		printV("prios", prios);
//		std::cout << st << std::endl;
	}

	std::cout << "To sample --------------------------------" << std::endl;
	std::vector<int> counts(num, 0);

	for (int i = 0; i < 1000; i ++) {
		auto rc = st.sample(4);
		auto indice = std::get<0>(rc);
		for (const auto& index: indice) {
			counts[index] ++;
		}
	}

//	printV("input", inputs);
	std::cout << st << std::endl;
	printV("counts", counts);
}

void test6() {
	std::random_device rd;
	std::mt19937 re(rd());
	std::uniform_real_distribution<> dist(0, 10.0);
	SegTree st(8);
	const int num = 8;
	float prio = 1;
	std::vector<float> inputs(num, 0);
	for(int i = 0; i < num; i ++) {
		inputs[i] = prio;
		if (prio > 1.5) {
			prio = 1;
		} else {
			prio = 2;
		}
	}

	for (int i = 0; i < num; i ++) {
		st.add(inputs[i]);

//		auto rc = st.sample(1);
//		auto indice = std::get<0>(rc);
//		auto prios = std::get<1>(rc);
//
//		printV("indice", indice);
//		printV("prios", prios);
//		std::cout << st << std::endl;
	}

	std::cout << "To sample --------------------------------" << std::endl;
	std::vector<int> counts(num, 0);

	for (int i = 0; i < 1000; i ++) {
		auto rc = st.sample(4);
		auto indice = std::get<0>(rc);
		for (const auto& index: indice) {
			counts[index] ++;
		}
	}

//	printV("input", inputs);
	std::cout << st << std::endl;
	printV("counts", counts);
}

void test7() {
	std::random_device rd;
	std::mt19937 re(rd());
	std::uniform_real_distribution<> dist(0, 10.0);
	SegTree st(8);
	const int num = 8;
	float prio = 0;
	std::vector<float> inputs(num, 0);
	for(int i = 0; i < num; i ++) {
		inputs[i] = prio;
		if (prio > 0.5) {
			prio = 0;
		} else {
			prio = 1;
		}
	}

	for (int i = 0; i < num; i ++) {
		st.add(inputs[i]);

//		auto rc = st.sample(1);
//		auto indice = std::get<0>(rc);
//		auto prios = std::get<1>(rc);
//
//		printV("indice", indice);
//		printV("prios", prios);
//		std::cout << st << std::endl;
	}

	std::cout << "To sample --------------------------------" << std::endl;
	std::vector<int> counts(num, 0);

	for (int i = 0; i < 1000; i ++) {
		auto rc = st.sample(4);
		auto indice = std::get<0>(rc);
		for (const auto& index: indice) {
			counts[index] ++;
		}
	}

//	printV("input", inputs);
	std::cout << st << std::endl;
	printV("counts", counts);
}

void test8() {
	const int num = 4096;
	const int batchSize = 64;

	std::random_device rd;
	std::mt19937 re(rd());
	std::uniform_real_distribution<> dist(0, 1.0);
	SegTree st(num);
//	float prio = (float)dist(re);

	for (int i = 0; i < num; i ++) {
		float addPrio = (float)dist(re);
		st.add(addPrio);

		if (i < batchSize) {
			continue;
		}

		auto rc = st.sample(batchSize);
		std::vector<int> indice = std::get<0>(rc);
		std::vector<float> prios = std::get<1>(rc);

		for (int j = 0; j < batchSize; j ++) {
			float updatePrio = (float)dist(re);
			prios[j] = updatePrio;
		}

		if ((i % 100) == 0) {
			std::cout << "updated " << i << std::endl;
			printV("updated indice: ", indice);
			printV("updated prios: ", prios);
		}
		st.update(indice, prios);
	}
}

void test9() {
	const int num = 8;
	SegTree st(num);

	for (int i = 0; i < 7; i ++) {
		st.add(1);
	}

	int index = st.sample(st.getSum());
	std::cout << "sampled index = " << index << std::endl;
}

void test10() {
	const int num = 32;
	SegTree st(num);

	for (int i = 0; i < num; i ++) {
		st.add(1);
	}
}
}

int main() {
	test10();
}
