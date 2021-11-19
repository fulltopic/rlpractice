/*
 * testptr.cpp
 *
 *  Created on: Nov 13, 2021
 *      Author: zf
 */

#include <iostream>
#include <string>
#include <memory>

namespace {
class A {
public:
	virtual ~A() = 0;
	virtual void print() = 0;
};
A::~A() {

}

template<typename DataType>
class B: public A {
private:
	DataType data;
public:
	B(DataType d): data(d) {}

	virtual ~B() {}
	virtual void print() {
		std::cout << "Print " << data << std::endl;
	}
};


std::shared_ptr<A> create() {
	return std::shared_ptr<A>(new B<int>(3));
}

enum Test {
	One = 1,
	Two = 2,
};

void testEnum() {
	int cmd = 1;

	switch (cmd) {
	case One:
		std::cout << "One " << std::endl;
		break;
	case Two:
		std::cout << "Two " << std::endl;
		break;
	default:
		std::cout << "Unexpected " << std::endl;
	}
}
}


int main() {
//	std::shared_ptr<A> test = create();
//	test->print();

	testEnum();
	return 0;
}

