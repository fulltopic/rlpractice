/*
 * envutils.h
 *
 *  Created on: Apr 2, 2021
 *      Author: zf
 */

#ifndef INC_GYMTEST_ENV_ENVUTILS_H_
#define INC_GYMTEST_ENV_ENVUTILS_H_

#include <type_traits>
#include <vector>
#include <stdint.h>

class EnvUtils {
public:
	static std::vector<float> FlattenVector(std::vector<float> const &input);

	template<typename T>
	static std::enable_if_t<!std::is_same<T, bool>::value && !std::is_same<T, int64_t>::value, std::vector<float>>
	FlattenVector(std::vector<std::vector<T>> const &input) {
		    std::vector<float> output;

		    for (auto const &element : input)
		    {
		        auto sub_vector = FlattenVector(element);

		        //An alternative to push_back
		        output.reserve(output.size() + sub_vector.size());
		        output.insert(output.end(), sub_vector.cbegin(), sub_vector.cend());
		    }

		    return output;
	}


	template<typename T>
	static std::enable_if_t<std::is_same<T, bool>::value, std::vector<bool>>
	FlattenVector(std::vector<std::vector<T>> const &input) {
		    std::vector<bool> output;

		    for (auto const &element : input)
		    {
		//	        auto sub_vector = Flatten_vector(element);

		        //An alternative to push_back
		        output.reserve(output.size() + element.size());
		        output.insert(output.end(), element.cbegin(), element.cend());
		    }

		    return output;
	}

	template<typename T>
	static std::enable_if_t<std::is_same<T, int64_t>::value, std::vector<int64_t>>
	FlattenVector(std::vector<std::vector<T>> const &input) {
		    std::vector<int64_t> output;

		    for (auto const &element : input)
		    {
		//	        auto sub_vector = Flatten_vector(element);

		        //An alternative to push_back
		        output.reserve(output.size() + element.size());
		        output.insert(output.end(), element.cbegin(), element.cend());
		    }

		    return output;
	}

	static bool OneDone(const std::vector<bool>& doneMark);
};




#endif /* INC_GYMTEST_ENV_ENVUTILS_H_ */
