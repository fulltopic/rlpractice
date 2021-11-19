/*
 * a3cconfig.cpp
 *
 *  Created on: Nov 7, 2021
 *      Author: zf
 */



#include "a3c/a3cconfig.h"

const std::string A3CConfig::ServerIp = "127.0.0.1";
const int A3CConfig::ServerPort = 3333;
const int A3CConfig::CmdLen = 3;

//A3CConfig::Cmd operator()(int cmd) {
//	switch (cmd) {
//	case 1:
//		return A3CConfig::AddGrad;
//	case 2:
//		return A3CConfig::SyncTarget;
//	case 10:
//		return A3CConfig::Start;
//	case 20:
//		return A3CConfig::Done;
//	case 100:
//		return A3CConfig::Invalid;
//	default:
//		return A3CConfig::Invalid;
//	}
//}


