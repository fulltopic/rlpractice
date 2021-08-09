/*
 * ProbeEnv2.h
 *
 *  Created on: Aug 8, 2021
 *      Author: zf
 */

#ifndef INC_PROBEENVS_PROBEENV2_H_
#define INC_PROBEENVS_PROBEENV2_H_


#include "ProbeEnv.h"
#include <vector>
#include <random>

class ProbeEnv2: public ProbeEnv{
public:
	ProbeEnv2(int len);
	virtual ~ProbeEnv2() = default;

	virtual std::vector<float>
		reset();
	virtual std::tuple<std::vector<float>, float, bool>
//	virtual std::vector<float>
		step(const int action, const bool render = false);

	virtual int getInputLen();

protected:
	const int inputLen;
	std::vector<float> state;

	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution = std::uniform_int_distribution<int>(0, 1);

//	float getDice();
	void updateState();
};




#endif /* INC_PROBEENVS_PROBEENV2_H_ */
