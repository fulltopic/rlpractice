/*
 * ProbeEnv4.h
 *
 *  Created on: Aug 8, 2021
 *      Author: zf
 */

#ifndef INC_PROBEENVS_PROBEENV4_H_
#define INC_PROBEENVS_PROBEENV4_H_


#include "ProbeEnv.h"

class ProbeEnv4: public ProbeEnv {
public:
	ProbeEnv4(int len);
	virtual ~ProbeEnv4() = default;

	virtual std::vector<float>
		reset();
	virtual std::tuple<std::vector<float>, float, bool>
		step(const int action, const bool render = false);

	virtual int getInputLen();

protected:
	const int inputLen;
};



#endif /* INC_PROBEENVS_PROBEENV4_H_ */
