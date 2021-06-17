/*
 * lmdbseqreader.cpp
 *
 *  Created on: Mar 23, 2021
 *      Author: zf
 */

#include "dbtools/lmdbseqreader.h"

#include <log4cxx/logger.h>

#include <exception>
#include <vector>
#include <cstring>

#include <caffe2/proto/caffe2_pb.h>
#include <caffe2/core/blob_serialization.h>


namespace {
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("lmdb"));
}

const std::string LmdbSeqReader::GetDbType() {
	return "lmdb";
}

LmdbSeqReader::LmdbSeqReader(const std::string path): dbPath(path),
		db(caffe2::db::CreateDB(GetDbType(), dbPath, caffe2::db::READ)),
		intDataOptions(), floatDataOptions(){
	intDataOptions = intDataOptions.dtype(c10::ScalarType::Int);
	floatDataOptions = floatDataOptions.dtype(c10::ScalarType::Float);

	cursor = db->NewCursor();
	LOG4CXX_INFO(logger, "Opened db " << dbPath);
}

LmdbSeqReader::~LmdbSeqReader() {
	try {
		db->Close();
		db.reset();
	} catch (std::exception& e) {
		LOG4CXX_ERROR(logger, "Failed to close db: " << e.what());
	}
}

bool LmdbSeqReader::hasNext() {
	return cursor->Valid();
}

void LmdbSeqReader::reset() {
	cursor->SeekToFirst();
}

//TODO: lmdb to tensor should be no copy
torch::Tensor LmdbSeqReader::proto2IntTensor(const caffe2::TensorProto& proto) {
	auto data = proto.int32_data();
	auto dims = proto.dims();
	auto vecDims = std::vector<int64_t>(dims.begin(), dims.end());

	torch::Tensor tensor = torch::zeros(vecDims, intDataOptions);
	std::memcpy(tensor.data_ptr(), static_cast<const void*>(data.data()), tensor.nbytes());

	return tensor;
}

torch::Tensor LmdbSeqReader::proto2FloatTensor(const caffe2::TensorProto& proto) {
	auto data = proto.float_data();
	auto dims = proto.dims();
	auto vecDims = std::vector<int64_t>(dims.begin(), dims.end());

	torch::Tensor tensor = torch::zeros(vecDims, floatDataOptions);
	std::memcpy(tensor.data_ptr(), static_cast<const void*>(data.data()), tensor.nbytes());

	return tensor;
}

torch::Tensor LmdbSeqReader::proto2IntTensor(const caffe2::TensorProto& proto, int seqLen) {
	auto data = proto.int32_data();
	auto dims = proto.dims();
	auto vecDims = std::vector<int64_t>(dims.begin(), dims.end());
	auto len = dims[0];
	vecDims[0] = seqLen;

	//TODO: Remove bound to assumed memory layout
	int64_t gap = 1;
	for (int i = 1; i < vecDims.size(); i ++) {
		gap *= vecDims[i];
	}

	torch::Tensor tensor = torch::zeros(vecDims, intDataOptions);

	if (len >= seqLen) {
		//Make final reward backpropa
		std::memcpy(tensor.data_ptr(), static_cast<const void*>(data.data() + (len- seqLen) * gap), tensor.nbytes());
	} else {
		std::memcpy(tensor.data_ptr(), static_cast<const void*>(data.data()), data.size() * sizeof(int32_t));
	}

	return tensor;
}

torch::Tensor LmdbSeqReader::proto2FloatTensor(const caffe2::TensorProto& proto, int seqLen) {
	auto data = proto.float_data();
	auto dims = proto.dims();
	auto vecDims = std::vector<int64_t>(dims.begin(), dims.end());
	auto len = dims[0];
	vecDims[0] = seqLen;

	int64_t gap = 1;
	for (int i = 1; i < vecDims.size(); i ++) {
		gap *= vecDims[i];
	}

	torch::Tensor tensor = torch::zeros(vecDims, floatDataOptions);
	if (len >= seqLen) {
		std::memcpy(tensor.data_ptr(), static_cast<const void*>(data.data() + (len - seqLen) * gap), tensor.nbytes());
	} else {
		std::memcpy(tensor.data_ptr(), static_cast<const void*>(data.data()), data.size() * sizeof(float));
	}
//	std::memcpy(tensor.data_ptr(), static_cast<const void*>(data.data()), tensor.nbytes());

	return tensor;
}

SeqData LmdbSeqReader::next() {
	const auto key = cursor->key();
	LOG4CXX_DEBUG(logger, "reader key: " << key);

	caffe2::TensorProtos protos;
	protos.ParseFromString(cursor->value());
	LOG4CXX_DEBUG(logger, "proto size: " << protos.protos_size());

	auto stateTensor = proto2IntTensor(protos.protos(0));
//	auto stateProto = protos.protos(0);
//	auto stateData = stateProto.int32_data();
//	auto stateDims = stateProto.dims();
//	std::vector<int64_t> stateVecDims = std::vector<int64_t>(stateDims.begin(), stateDims.end());
//	torch::Tensor stateTensor = torch::zeros(stateVecDims, dataTypeOptions);
//	std::memcpy(stateTensor.data_ptr(), static_cast<const void*>(stateData.data()), stateTensor.nbytes());

//	LOG4CXX_DEBUG(logger, "dims size: " << stateDims.size());
//	LOG4CXX_DEBUG(logger, "proto dims: " << stateVecDims);
//	LOG4CXX_DEBUG(logger, "vecData: " << stateVecData);
//	LOG4CXX_DEBUG(logger, "tensor sizes: " << stateTensor.numel() << ", " << stateTensor.nbytes());
	LOG4CXX_DEBUG(logger, "close tiles: " << stateTensor[0][0]);
//	LOG4CXX_DEBUG(logger, "zero tensor size: " << zeroTensor.numel() << ", " << zeroTensor.nbytes());
//	LOG4CXX_DEBUG(logger, "zero close tiles: " << zeroTensor[0][0]);

	auto actionTensor = proto2IntTensor(protos.protos(1));
	LOG4CXX_DEBUG(logger, "actions: " << actionTensor);

	auto rewardTensor = proto2FloatTensor(protos.protos(2));
	LOG4CXX_DEBUG(logger, "reward: " << rewardTensor);

	cursor->Next();

	return {stateTensor, actionTensor, rewardTensor};
}

SeqData LmdbSeqReader::nextWithSeq(int seqLen) {
	const auto key = cursor->key();
	LOG4CXX_DEBUG(logger, "reader key: " << key);

	caffe2::TensorProtos protos;
	protos.ParseFromString(cursor->value());
	LOG4CXX_DEBUG(logger, "proto size: " << protos.protos_size());

	auto stateTensor = proto2IntTensor(protos.protos(0), seqLen);
	LOG4CXX_DEBUG(logger, "close tiles: " << stateTensor[0][0]);

	auto actionTensor = proto2IntTensor(protos.protos(1), seqLen);
	LOG4CXX_DEBUG(logger, "actions: " << actionTensor);

	auto rewardTensor = proto2FloatTensor(protos.protos(2), seqLen);
	LOG4CXX_DEBUG(logger, "reward: " << rewardTensor);

	cursor->Next();

	return {stateTensor, actionTensor, rewardTensor};
}

std::vector<SeqData> LmdbSeqReader::next(int batchSize) {
	std::vector<SeqData> data;
	for (int i = 0; i < batchSize; i ++) {
		if (hasNext()) {
			data.push_back(next());
		} else {
			break;
		}
	}

	return data;
}


std::vector<SeqData> LmdbSeqReader::nextWithSeq(int batchSize, int seqLen) {
	std::vector<SeqData> data;
	for (int i = 0; i < batchSize; i ++) {
		if (hasNext()) {
			data.push_back(nextWithSeq(seqLen));
		} else {
			break;
		}
	}

	return data;
}

