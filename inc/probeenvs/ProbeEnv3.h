/*
 * ProbeEnv3.h
 *
 *  Created on: Aug 8, 2021
 *      Author: zf
 */

#ifndef INC_PROBEENVS_PROBEENV3_H_
#define INC_PROBEENVS_PROBEENV3_H_

#include "ProbeEnv.h"

class ProbeEnv3: public ProbeEnv {
public:
	ProbeEnv3(int len);
	virtual ~ProbeEnv3() = default;

	virtual std::vector<float>
		reset();
	virtual std::tuple<std::vector<float>, float, bool>
//	virtual std::vector<float>
		step(const int action, const bool render = false);

	virtual int getInputLen();

protected:
	const int inputLen;
	std::vector<float> state;

//	float getDice();
	void updateState();
};


#endif /* INC_PROBEENVS_PROBEENV3_H_ */
