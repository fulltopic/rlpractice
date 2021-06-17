/*
 * communicator.h
 *
 *  Created on: Apr 2, 2021
 *      Author: Omegastick
 *      From https://github.com/Omegastick/pytorch-cpp-rl
 *
 */

#ifndef INC_GYMTEST_ENV_COMMUNICATOR_H_
#define INC_GYMTEST_ENV_COMMUNICATOR_H_


#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include <log4cxx/logger.h>

#include <msgpack.hpp>


#include "requests.h"
#include <zmq.hpp>

class Communicator
{
  public:
    Communicator(const std::string &url);
    ~Communicator();

    std::string get_raw_response();

    template <class T>
    std::unique_ptr<T> get_response()
    {
        // Receive message
        zmq::message_t packed_msg;
        bool rcved = socket->recv(&packed_msg);
        if (rcved) {
        	LOG4CXX_DEBUG(logger, "Communicator received " << rcved);
        } else {
        	LOG4CXX_ERROR(logger, "Communicator received " << rcved);
        }
//        LOG4CXX_INFO(logger, "msg size: " << packed_msg.size());

        // Desrialize message
        msgpack::object_handle object_handle = msgpack::unpack(static_cast<char *>(packed_msg.data()), packed_msg.size());
        msgpack::object object = object_handle.get();

        // Fill out response object
        std::unique_ptr<T> response = std::make_unique<T>();
        try
        {
            object.convert(response);
        }
        catch (...)
        {
        	LOG4CXX_ERROR(logger, "Communication error: " << object);
        }

        return response;
    }

    template <class T>
    void send_request(const Request<T> &request)
    {
        msgpack::sbuffer buffer;
        msgpack::pack(buffer, request);

        zmq::message_t message(buffer.size());
        std::memcpy(message.data(), buffer.data(), buffer.size());
        socket->send(message);
    }

  private:
	log4cxx::LoggerPtr logger;

    std::unique_ptr<zmq::context_t> context;
    std::unique_ptr<zmq::socket_t> socket;
};



#endif /* INC_GYMTEST_ENV_COMMUNICATOR_H_ */
