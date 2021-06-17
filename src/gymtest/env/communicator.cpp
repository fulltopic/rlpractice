/*
 * communicator.cpp
 *
 *  Created on: Apr 2, 2021
 *      Author: zf
 */


#include <memory>
#include <string>


#include "gymtest/env/communicator.h"
#include "gymtest/env/requests.h"

#include <zmq.hpp>


Communicator::Communicator(const std::string &url)
	: logger(log4cxx::Logger::getLogger("Communicator_" + url))
{
    context = std::make_unique<zmq::context_t>(1);
    socket = std::make_unique<zmq::socket_t>(*context, ZMQ_PAIR);

    LOG4CXX_INFO(logger, "To connect to " << url);
    socket->connect(url.c_str());
    LOG4CXX_INFO(logger, get_raw_response());
}

Communicator::~Communicator() {}

std::string Communicator::get_raw_response()
{
    // Receive message
    zmq::message_t zmq_msg;
    bool recved = socket->recv(&zmq_msg);
    if (recved) {
    	LOG4CXX_INFO(logger, "received: " << recved << ", " << zmq_msg.size());
    } else {
    	LOG4CXX_ERROR(logger, "received: " << recved);
    }

    // Cast message to string
    std::string response = std::string(static_cast<char *>(zmq_msg.data()), zmq_msg.size());

    return response;
}


