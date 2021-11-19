/*
 * dummyfuncs.h
 *
 *  Created on: Nov 12, 2021
 *      Author: zf
 */

#ifndef INC_A3C_DUMMYFUNCS_H_
#define INC_A3C_DUMMYFUNCS_H_

#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

#include <torch/torch.h>

#include <cstddef>

class DummyFuncs {
private:
	DummyFuncs();
	static log4cxx::LoggerPtr logger; // = log4cxx::Logger::getLogger("a3cdummy");

public:

	DummyFuncs(const DummyFuncs&) = delete;
	DummyFuncs operator=(const DummyFuncs&) = delete;
	~DummyFuncs() = default;

//	static DummyFuncs& GetInstance();

	static void DummyRcv(void* data, std::size_t len);
	static bool TorchSizesEq(const torch::IntArrayRef& a, const torch::IntArrayRef& b);
};



#endif /* INC_A3C_DUMMYFUNCS_H_ */
