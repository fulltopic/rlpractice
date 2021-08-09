/*
 * ProbeEnv1.h
 *
 *  Created on: Aug 7, 2021
 *      Author: zf
 */

#ifndef INC_PROBEENVS_PROBEENV1_H_
#define INC_PROBEENVS_PROBEENV1_H_

#include "ProbeEnv.h"
#include <vector>

class ProbeEnv1: public ProbeEnv{
public:
	ProbeEnv1(int len);
	virtual ~ProbeEnv1() = default;

	virtual std::vector<float>
		reset();
	virtual std::tuple<std::vector<float>, float, bool>
//	virtual std::vector<float>
		step(const int action, const bool render = false);

	virtual int getInputLen();

protected:
	const int inputLen;
};


#endif /* INC_PROBEENVS_PROBEENV1_H_ */
