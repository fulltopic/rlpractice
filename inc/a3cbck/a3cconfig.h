/*
 * a3cconfig.h
 *
 *  Created on: Nov 6, 2021
 *      Author: zf
 */

#ifndef INC_A3C_A3CCONFIG_H_
#define INC_A3C_A3CCONFIG_H_

#include <string>

class A3CConfig {
public:
	enum Cmd {
		AddGrad = 1,
		SyncTarget = 2,

		Start = 10,
		Done = 20,

		Invalid = 100,
	};

	static const int CmdLen;
	//data command: {Cmd, StartIndex, EndIndex(inclusize)}
//	A3CConfig::Cmd operator()(int cmd);

	static const std::string ServerIp;
	static const int ServerPort;

	static constexpr int BufCap = 32768;
};



#endif /* INC_A3C_A3CCONFIG_H_ */
