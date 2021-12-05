############################ Tensorboard ##################################
set(TENSORBOARD_PATH /home/zf/workspaces/workspace_cpp/tensorboard_logger/)
set(TENSORBOARD_INC_PATH ${TENSORBOARD_PATH}/include)
set(tensorboard_libs ${TENSORBOARD_PATH}/build/libtensorboard_logger.a)
include_directories(${TENSORBOARD_INC_PATH})
include_directories(${TENSORBOARD_PATH}/build/)

add_library(a3c OBJECT 
								a3ctcpclientconn.cpp
								a3ctcpserver.cpp
								a3ctcpconfig.cpp
								a3ctcpserverconn.cpp
								dummyfuncs.cpp
								a3ctcpserverhandleinterface.cpp
								a3ctcphandlefactory.cpp
								a3ctcpmsghds.cpp
								a3cgradque.cpp
							)
