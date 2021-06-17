/*
 * testcpp.cpp
 *
 *  Created on: Apr 30, 2021
 *      Author: zf
 */

#include <vector>
#include <queue>
#include <iostream>
#include <string>

namespace {
template<typename T>
void printVector(const std::vector<T>& datas, std::string cmt) {
	std::cout << cmt << std::endl;

	for (const auto& data: datas) {
		std::cout << data << ", " ;
	}
	std::cout << std::endl;
}
void test0() {
	std::queue<std::vector<float>> q;

	std::vector<float> v0{1, 2, 3, 4};
	q.push(v0);

	for (int i = 0; i < 10; i ++) {
		std::vector<float> v1 = std::vector<float>(4, i);
		v0 = v1;
		printVector(v0, "v0");
	}
	std::vector<float> v2 = q.front();
	q.pop();

	printVector(v2, "popped v");
}
}


int main() {
	test0();
}

