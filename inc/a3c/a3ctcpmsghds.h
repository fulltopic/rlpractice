/*
 * a3ctcpmsghds.h
 *
 *  Created on: Nov 18, 2021
 *      Author: zf
 */

#ifndef INC_A3C_A3CTCPMSGHDS_H_
#define INC_A3C_A3CTCPMSGHDS_H_

#include <iostream>
#include <string>
#include <vector>
#include <memory>


#include "a3ctcpconfig.h"
#include "a3ctcpmsghd.h"



//struct GradSyncReq {
//	A3CTCPCmdHd hd;
//};
using GradSyncReq = A3CTCPCmdHd;

struct GradUpdateReq{
	A3CTCPCmdHd hd;
	uint64_t index = 0;
};

struct GradUpdateRspHd {
	A3CTCPCmdHd hd;
	uint64_t index = 0;
	uint64_t dataLen = 0;
};

//struct GradUpdateError {
//	A3CTCPCmdHd hd;
//};
using GradUpdateError = A3CTCPCmdHd;

using GradUpdateComplete = A3CTCPCmdHd;
//
//struct GradComplete {
//	A3CTCPCmdHd hd;
//};
//
//struct TargetSyncStartReq {
//	A3CTCPCmdHd hd;
//};
using TargetSyncStartReq = A3CTCPCmdHd;
using TargetSyncStartRsp = A3CTCPCmdHd;

struct TargetSyncReq {
	A3CTCPCmdHd hd;
	uint64_t index;
};

struct TargetSyncRspHd {
	A3CTCPCmdHd hd;
	uint64_t index;
	uint64_t sndLen;
};

//struct TargetSyncError {
//	A3CTCPCmdHd hd;
//};
using TargetSyncError = A3CTCPCmdHd;
using TargetSyncComplete = A3CTCPCmdHd;

using TestMsg = A3CTCPCmdHd;

class A3CTCPCmdFactory {
public:
	static GradSyncReq* CreateGradSyncReq (char* data);

	static GradUpdateReq* CreateGradUpdateReq (char* data, uint64_t index);

	static GradUpdateRspHd* CreateGradUpdateRspHd (char* data, uint64_t index, uint64_t sndLen);

	static GradUpdateError* CreateGradUpdateError (char* data);

	static GradUpdateComplete* CreateGradComplete (char* data);

	static TargetSyncStartReq* CreateTargetSyncStartReq (char* data);

	static TargetSyncStartRsp* CreateTargetSyncStartRsp (char* data);

	static TargetSyncReq* CreateTargetSyncReq (char* data, uint64_t index);

	static TargetSyncRspHd* CreateTargetSyncRspHd (char* data, uint64_t index, uint64_t sndLen);

	static TargetSyncComplete* CreateTargetSyncComplete (char* data);

	static TargetSyncError* CreateTargetSyncError (char* data);

	static TestMsg* CreateTest (char* data);
};

#endif /* INC_A3C_A3CTCPMSGHDS_H_ */
