/*
 * envutils.cpp
 *
 *  Created on: Apr 2, 2021
 *      Author: zf
 */

#include "gymtest/env/envutils.h"

//std::vector<float> EnvUtils::flattenVector(std::vector<float> const &input) {
//	return input;
//}
//
//std::vector<float> EnvUtils::flattenVector(std::vector<std::vector<float>> const &input) {
//    std::vector<float> output;
//
//    for (auto const &element : input)
//    {
//        auto sub_vector = flattenVector(element);
//
//        //An alternative to push_back
//        output.reserve(output.size() + sub_vector.size());
//        output.insert(output.end(), sub_vector.cbegin(), sub_vector.cend());
//    }
//
//    return output;
//}
//
//
//std::vector<bool> EnvUtils::flattenVector(std::vector<bool> const &input) {
//	return input;
//}
//
//std::vector<bool> EnvUtils::flattenVector(std::vector<std::vector<bool>> const &input) {
//    std::vector<bool> output;
//
//    for (auto const &element : input)
//    {
//        auto sub_vector = flattenVector(element);
//
//        //An alternative to push_back
//        output.reserve(output.size() + sub_vector.size());
//        output.insert(output.end(), sub_vector.cbegin(), sub_vector.cend());
//    }
//
//    return output;
//}

std::vector<float> EnvUtils::FlattenVector(std::vector<float> const &input) {
	return input;
}

//template<typename T>
//static std::vector<float> FlattenVector(std::vector<std::vector<T>> const &input) {
//    std::vector<float> output;
//
//    for (auto const &element : input)
//    {
//        auto sub_vector = flatten_vector(element);
//
//        //An alternative to push_back
//        output.reserve(output.size() + sub_vector.size());
//        output.insert(output.end(), sub_vector.cbegin(), sub_vector.cend());
//    }
//
//    return output;
//}

bool EnvUtils::OneDone(const std::vector<bool>& doneMark) {
	for (const auto& done: doneMark) {
		if (done) {
			return true;
		}
	}
	return false;
}
