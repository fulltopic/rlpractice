/*
 * testptrbase.cpp
 *
 *  Created on: Aug 7, 2021
 *      Author: zf
 */



#include <vector>
#include <memory>
#include <string>
#include <iostream>

namespace {
class Base {
public:
	Base() = default;
	virtual ~Base() = default;

	virtual void test() = 0;
};

class Derive1: public Base {
public:
	Derive1() = default;
	virtual ~Derive1() = default;

	virtual void test() {
		std::cout << "d test 1" << std::endl;
	}
};

class Derive2:public Base {
public:
	Derive2() = default;
	virtual ~Derive2() = default;

	virtual void test() {
		std::cout << "d test 2" << std::endl;
	}
};

void test0() {
	std::vector<std::unique_ptr<Base>> vec;
	vec.push_back(std::make_unique<Derive1>());
	vec.push_back(std::make_unique<Derive2>());

	for (int i = 0; i < vec.size(); i ++) {
		vec[i]->test();
	}
}
}

int main() {
	test0();
}
