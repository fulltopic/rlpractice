/*
 * a3ctcpconfig.h
 *
 *  Created on: Nov 9, 2021
 *      Author: zf
 */

#ifndef INC_A3C_A3CTCPCONFIG_H_
#define INC_A3C_A3CTCPCONFIG_H_

#include <string>

class A3CTCPConfig {
public:
	enum Cmd {
		AddGrad = 1,
		SyncTarget = 2,

		StartGrad = 6,
		GradSending = 7,
		EndGrad = 8,
		GradInterrupt = 9,

		StartTarget = 16,
		TargetSending = 17,
		EndTarget = 18,
		TargetInterrupt = 19,

		Test = 100,

		Invalid = 200,
	};

	static const int CmdLen;
	//data command: {Cmd, StartIndex, EndIndex(inclusize)}
//	A3CConfig::Cmd operator()(int cmd);

	static const std::string ServerIp;
	static const int ServerPort;

	static constexpr int BufCap = 32768;
};




#endif /* INC_A3C_A3CTCPCONFIG_H_ */
