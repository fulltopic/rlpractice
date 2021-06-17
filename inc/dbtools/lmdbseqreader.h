/*
 * lmdbseqreader.h
 *
 *  Created on: Mar 23, 2021
 *      Author: zf
 */

#ifndef INC_DBTOOLS_LMDBSEQREADER_H_
#define INC_DBTOOLS_LMDBSEQREADER_H_

#include <string>
#include <vector>
#include <memory>

#include <torch/torch.h>

#include <caffe2/core/db.h>
#include <caffe2/core/init.h>
//#include <caffe2/proto/caffe2_legacy.pb.h>
//#include <caffe2/core/logging.h>

struct SeqData {
	torch::Tensor states;
	torch::Tensor actions;
	torch::Tensor rewards;
};

class LmdbSeqReader {
public:
	static const std::string GetDbType();

	explicit LmdbSeqReader(const std::string path);
	LmdbSeqReader(const LmdbSeqReader&) = delete;
	LmdbSeqReader& operator=(const LmdbSeqReader&) = delete;

	~LmdbSeqReader();

	bool hasNext();
	void reset();

	SeqData next();
	SeqData nextWithSeq(int seqLen);
//	void next();
	std::vector<SeqData> next(int batchSize);
	std::vector<SeqData> nextWithSeq(int batchSize, int seqLen);

private:
	std::string dbPath;

	torch::TensorOptions intDataOptions;
	torch::TensorOptions floatDataOptions;

	std::unique_ptr<caffe2::db::DB> db;
	std::unique_ptr<caffe2::db::Cursor> cursor;

	torch::Tensor proto2IntTensor(const caffe2::TensorProto& proto);
	torch::Tensor proto2FloatTensor(const caffe2::TensorProto& proto);

	torch::Tensor proto2IntTensor(const caffe2::TensorProto& proto, int seqLen);
	torch::Tensor proto2FloatTensor(const caffe2::TensorProto& proto, int seqLen);

};



#endif /* INC_DBTOOLS_LMDBSEQREADER_H_ */
