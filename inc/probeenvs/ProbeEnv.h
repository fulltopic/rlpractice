/*
 * ProbeEnv.h
 *
 *  Created on: Aug 7, 2021
 *      Author: zf
 */

#ifndef INC_PROBEENVS_PROBEENV_H_
#define INC_PROBEENVS_PROBEENV_H_

#include <string>
#include <vector>
#include <tuple>
#include <cstdlib>

class ProbeEnv {
public:
	ProbeEnv();
	virtual ~ProbeEnv();

	virtual std::vector<float> reset() = 0;
	virtual std::tuple<std::vector<float>, float, bool>
//	virtual std::vector<float>
		step(const int action, const bool render = false) = 0;
	virtual int getInputLen() = 0;
};



#endif /* INC_PROBEENVS_PROBEENV_H_ */
