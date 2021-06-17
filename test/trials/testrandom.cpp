/*
 * testrandom.cpp
 *
 *  Created on: Apr 10, 2021
 *      Author: zf
 */


#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>

int main()
{
    // Seed with a real random value, if available
    std::random_device r;

    // Choose a random mean between 1 and 6
    std::default_random_engine e1;
//    std::uniform_int_distribution<int> uniform_dist(0, 31);
//
//    for (int i = 0; i < 32; i ++) {
//    	int mean = uniform_dist(e1);
//    	std::cout << "random " << i << " = " << mean << '\n';
//    }
//    // Generate a normal distribution around that mean
//    std::seed_seq seed2{r(), r(), r(), r(), r(), r(), r(), r()};
//    std::mt19937 e2(seed2);
//    std::normal_distribution<> normal_dist(mean, 2);
//
//    std::map<int, int> hist;
//    for (int n = 0; n < 10000; ++n) {
//        ++hist[std::round(normal_dist(e2))];
//    }
//    std::cout << "Normal distribution around " << mean << ":\n";
//    for (auto p : hist) {
//        std::cout << std::fixed << std::setprecision(1) << std::setw(2)
//                  << p.first << ' ' << std::string(p.second/200, '*') << '\n';
//    }
//    std::cout << "float: " << sizeof(float) << ", int: " << sizeof(int) << std::endl;

	auto gen = std::mt19937(e1());
	std::vector<float> param{0.1, 0.1, 0.1, 0.7};
	std::discrete_distribution<int> d(param.begin(), param.end());
	std::vector<int> counts(param.size(), 0);

	for (int i = 0; i < 30; i ++) {
		auto value = d(gen);
		counts[value] ++;
		std::cout << "random " << i << " = " << value << std::endl;
	}
	for (int i = 0; i < counts.size(); i ++) {
		std::cout << "ratio" << i << " = " << (float)counts[i] / 30 << std::endl;
	}
}

