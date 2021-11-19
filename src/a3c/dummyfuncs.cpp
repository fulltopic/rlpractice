/*
 * dummyfuncs.cpp
 *
 *  Created on: Nov 12, 2021
 *      Author: zf
 */



#include "a3c/dummyfuncs.h"

DummyFuncs::DummyFuncs() {

}

log4cxx::LoggerPtr DummyFuncs::logger = log4cxx::Logger::getLogger("a3cdummy");

//DummyFuncs& DummyFuncs::GetInstance() {
//	static DummyFuncs instance;
//
//	return instance;
//}

void DummyFuncs::DummyRcv(void* data, std::size_t len) {
	LOG4CXX_ERROR(logger, "Dummy receive handle");
}

bool DummyFuncs::TorchSizesEq(const torch::IntArrayRef& a, const torch::IntArrayRef& b) {
	if (a.size() != b.size()) {
		return false;
	}

	for (int i = 0; i < a.size(); i ++) {
		if (a[i] != b[i]) {
			return false;
		}
	}

	return true;
}
