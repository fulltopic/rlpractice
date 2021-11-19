/*
 * a3ctcpmsghds.cpp
 *
 *  Created on: Nov 18, 2021
 *      Author: zf
 */



#include "a3c/a3ctcpmsghds.h"

GradSyncReq* A3CTCPCmdFactory::CreateGradSyncReq (char* data) {
	GradSyncReq* req = (GradSyncReq*)(data);
	req->cmd = A3CTCPConfig::StartGrad;
	req->expSize = sizeof(uint64_t) * 2;

	return req;
}

GradUpdateReq* A3CTCPCmdFactory::CreateGradUpdateReq(char* data, uint64_t index) {
	GradUpdateReq* req = (GradUpdateReq*)(data);
	req->hd.cmd = A3CTCPConfig::GradSending;
	req->index = index;
	req->hd.expSize = sizeof(uint64_t) + sizeof(A3CTCPCmdHd);

	return req;
}

GradUpdateRspHd* A3CTCPCmdFactory::CreateGradUpdateRspHd (char* data, uint64_t index, uint64_t sndLen) {
	GradUpdateRspHd* req = (GradUpdateRspHd*)(data);
	req->hd.cmd = A3CTCPConfig::GradSending;
	req->hd.expSize = sizeof(A3CTCPCmdHd) + sizeof(uint64_t) * 2 + sndLen;
	req->index = index;
	req->dataLen = sndLen;

	return req;
}

GradUpdateError* A3CTCPCmdFactory::CreateGradUpdateError (char* data) {
	GradUpdateError* req = (GradUpdateError*)(data);
	req->cmd = A3CTCPConfig::GradInterrupt;
	req->expSize = sizeof(A3CTCPCmdHd);

	return req;
}

GradUpdateComplete* A3CTCPCmdFactory::CreateGradComplete(char* data) {
	GradUpdateComplete* req = (GradUpdateComplete*)(data);
	req->cmd = A3CTCPConfig::EndGrad;
	req->expSize = sizeof(A3CTCPCmdHd);

	return req;
}

TargetSyncStartReq* A3CTCPCmdFactory::CreateTargetSyncStartReq(char* data){
	TargetSyncStartReq* req = (TargetSyncStartReq*)data;
	req->cmd = A3CTCPConfig::StartTarget;
	req->expSize = sizeof(A3CTCPCmdHd);

	return req;
}

TargetSyncStartRsp* A3CTCPCmdFactory::CreateTargetSyncStartRsp(char* data) {
	return CreateTargetSyncStartReq(data);
}

TargetSyncReq* A3CTCPCmdFactory::CreateTargetSyncReq(char* data, uint64_t index) {
	TargetSyncReq* req = (TargetSyncReq*)(data);
	req->hd.cmd = A3CTCPConfig::TargetSending;
	req->hd.expSize = sizeof(A3CTCPCmdHd) + sizeof(uint64_t);
	req->index = index;

	return req;
}

TargetSyncRspHd* A3CTCPCmdFactory::CreateTargetSyncRspHd(char* data, uint64_t index, uint64_t sndLen) {
	TargetSyncRspHd* rsp = (TargetSyncRspHd*)(data);
	rsp->hd.cmd = A3CTCPConfig::TargetSending;
	rsp->hd.expSize = sizeof(A3CTCPCmdHd) + sizeof(uint64_t) * 2 + sndLen;
	rsp->index = index;
	rsp->sndLen = sndLen;

	return rsp;
}

TargetSyncComplete* A3CTCPCmdFactory::CreateTargetSyncComplete(char* data) {
	TargetSyncComplete* req = (TargetSyncComplete*)(data);
	req->cmd = A3CTCPConfig::EndTarget;
	req->expSize = sizeof(A3CTCPCmdHd);

	return req;
}

TargetSyncError* A3CTCPCmdFactory::CreateTargetSyncError(char* data) {
	TargetSyncError* req = (TargetSyncError*)(data);
	req->cmd = A3CTCPConfig::TargetInterrupt;
	req->expSize = sizeof(A3CTCPCmdHd);

	return req;
}


TestMsg* A3CTCPCmdFactory::CreateTest(char* data) {
	TestMsg* req = (TestMsg*)(data);
	req->cmd = A3CTCPConfig::Test;
	req->expSize = sizeof(A3CTCPCmdHd);

	return req;
}

