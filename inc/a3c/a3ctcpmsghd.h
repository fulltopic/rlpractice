/*
 * a3ctcpmsghd.h
 *
 *  Created on: Nov 19, 2021
 *      Author: zf
 */

#ifndef INC_A3C_A3CTCPMSGHD_H_
#define INC_A3C_A3CTCPMSGHD_H_

#include "a3ctcpconfig.h"

struct A3CTCPCmdHd {
	uint64_t cmd;
	uint64_t expSize;
};



#endif /* INC_A3C_A3CTCPMSGHD_H_ */
