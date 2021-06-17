/*
 * priorb.cpp
 *
 *  Created on: May 23, 2021
 *      Author: zf
 */


#include "gymtest/utils/priorb.h"
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>

#include <torch/torch.h>

PrioRb::PrioRb(int cap): capacity(cap),
	states(cap, std::vector<float>()),
	nextStates(cap, std::vector<float>()),
	rewards(cap, 0),
	dones(cap, false),
	actions(cap, 0),
	prioSumTree(cap * 2, 0),
	minHeap(cap, 0),
	prio2MinHeap(cap, -1)
{
	for (int i = 0; i < minHeap.size(); i ++) {
		minHeap[i] = i;
	}
}

void PrioRb::put(std::vector<float> state, std::vector<float> nextState,
	std::vector<float> reward, std::vector<long> action, std::vector<bool> done, const int batchSize) {
	const auto stateSize = state.size() / batchSize;
//	const auto rewardSize = reward.size() / batchSize;
//	const auto actionSize = action.size() / batchSize;
//	const auto doneSize = done.size() / batchSize;

	for (int i = 0; i < batchSize; i ++) {
		states[index].resize(stateSize);
		std::copy(state.begin() + i * stateSize, state.begin() + (i + 1) * stateSize, states[index].begin());
		nextStates[index].resize(stateSize);
		std::copy(nextState.begin() + i * stateSize, nextState.begin() + (i + 1) * stateSize, nextStates[index].begin());

		rewards[index] = reward[i];
		actions[index] = action[i];
		dones[index] = done[i];

//		prioSumTree[index + capacity] = defaultMaxPrio;
		updatePrio(index, defaultMaxPrio);

		index = (index + 1) % capacity;
	}

	count = std::min(count + batchSize, capacity);
}

int PrioRb::searchSumTree(float sumPrefix) {
	int index = 1;
	const float origPrefix = sumPrefix;
	std::vector<int> path;
	path.push_back(index);

	while (index * 2 < (capacity + count)) {
		int left = index * 2;
		int right = left + 1;

		if (sumPrefix <= prioSumTree[left]) {
			index = left;
			path.push_back(index);

		} else if (right < (capacity + count)) {
			sumPrefix -= prioSumTree[left];
			index = right;
			path.push_back(index);
		}
		else {
			index = left;
			path.push_back(index);

			break;
		}
	}

	//TODO: seemed some float uncertainty
	if ((index >= (capacity + count)) || (index < capacity)) {
		LOG4CXX_ERROR(logger, "invalid index " << index << " = " << origPrefix << ", " << sumPrefix);
		LOG4CXX_ERROR(logger, "path: " << path);
		index = capacity + count - 1;
	}
	return index;
}

void PrioRb::updateSumTree(const int index, const float prio) {
//	LOG4CXX_DEBUG(logger, "Update sumTree " << index << " into " << prio);

	int sumindex = index + capacity;
	float delta = prio - prioSumTree[sumindex];
//	LOG4CXX_DEBUG(logger, "delta = " << delta);

	while (sumindex > 0) {
		prioSumTree[sumindex] += delta;
		sumindex /= 2;
	}

//	LOG4CXX_DEBUG(logger, "Updated into " << prioSumTree);
}

float PrioRb::getMinW() {
	int index = minHeap[0];
	index += capacity;

	return prioSumTree[index];
}

float PrioRb::getSum() {
	return prioSumTree[1];
}

void PrioRb::updateMinHeap(const int index) {
//	LOG4CXX_DEBUG(logger, "update heap " << index);

	const static int NONE = 0;
	const static int UP = 1;
	const static int DOWN = 2;

	int move = NONE;
	const int prioIndex = index;
	if (prio2MinHeap[prioIndex] < 0) {
		prio2MinHeap[prioIndex] = index;
	}
	int heapIndex = prio2MinHeap[prioIndex];

	int leftHeapIndex = heapIndex * 2 + 1;
	if (leftHeapIndex < count) {
		int leftPrioIndex = minHeap[leftHeapIndex];
		if (prioSumTree[leftPrioIndex + capacity] < prioSumTree[prioIndex + capacity]) {
			move = DOWN;
		} else {
			int rightHeapIndex = heapIndex * 2 + 2;
			if (rightHeapIndex < count) {
				int rightPrioIndex = minHeap[rightHeapIndex];
				if (prioSumTree[rightPrioIndex + capacity] < prioSumTree[prioIndex + capacity]) {
					move = DOWN;
				}
			}
		}
	}
	if (move == NONE) {
		if (heapIndex > 0) {
			int parentHeapIndex = (heapIndex - 1) / 2;
			int parentPrioIndex = minHeap[parentHeapIndex];
			if (prioSumTree[parentPrioIndex + capacity] > prioSumTree[prioIndex + capacity]) {
				move = UP;
			}
		}
	}

	if (move == UP) {
		while (heapIndex > 0) {
			int parentHeapIndex = (heapIndex - 1) / 2;
			int parentPrioIndex = minHeap[parentHeapIndex];
			if (prioSumTree[parentPrioIndex + capacity] > prioSumTree[prioIndex + capacity]) {
				minHeap[parentHeapIndex] = prioIndex;
				minHeap[heapIndex] = parentPrioIndex;
				prio2MinHeap[prioIndex] = parentHeapIndex;
				prio2MinHeap[parentPrioIndex] = heapIndex;

				heapIndex = parentHeapIndex;
			} else {
				break;
			}
		}
	} else if (move == DOWN) {
		while (heapIndex < count) {
			int leftHeapIndex = heapIndex * 2 + 1;
			int rightHeapIndex = leftHeapIndex + 1;
			if (leftHeapIndex < count) {
				int childHeapIndex = -1;

				int leftPrioIndex = minHeap[leftHeapIndex];
				if (prioSumTree[leftPrioIndex + capacity] < prioSumTree[prioIndex + capacity]) {
					childHeapIndex = leftHeapIndex;
				}
				if (rightHeapIndex < count) {
					int rightPrioIndex = minHeap[rightHeapIndex];
					if ((childHeapIndex >= 0) && (prioSumTree[rightPrioIndex + capacity] < prioSumTree[leftPrioIndex + capacity])) {
						childHeapIndex = rightHeapIndex;
					} else if ((childHeapIndex < 0) && (prioSumTree[rightPrioIndex + capacity] < prioSumTree[prioIndex + capacity])) {
						childHeapIndex = rightHeapIndex;
					}
				}

				if (childHeapIndex >= 0) {
					int childPrioIndex = minHeap[childHeapIndex];
					minHeap[heapIndex] = childPrioIndex;
					minHeap[childHeapIndex] = prioIndex;
					prio2MinHeap[prioIndex] = childHeapIndex;
					prio2MinHeap[childPrioIndex] = heapIndex;

					heapIndex = childHeapIndex;
				} else {
					break;
				}
			} else {
				break;
			}
		}
	}

//	LOG4CXX_DEBUG(logger, "after update heap: " << minHeap);
//	LOG4CXX_DEBUG(logger, "after update map: " << prio2MinHeap);
}

std::tuple<std::vector<int>, std::vector<float>>
PrioRb::sampleBatch(int batchSize) {
	torch::Tensor batchSumTensor = torch::rand({batchSize}) * getSum();
	float* batchSum = batchSumTensor.data_ptr<float>();
	std::vector<int> indices(batchSize, 0);
	std::vector<float> prios(batchSize, 0);

	for (int i = 0; i < batchSize; i ++) {
		int index = searchSumTree(batchSum[i]);
		indices[i] = index - capacity;
		prios[i] = prioSumTree[index];
	}

	return {indices, prios};
}

void PrioRb::updatePrio(const int index, const float prio) {
	LOG4CXX_DEBUG(logger, "update prio " << index << " = " << prio);
	updateSumTree(index, prio);
	updateMinHeap(index);
}

void PrioRb::updatePrios(const std::vector<int>& indices, const std::vector<float>& prios) {
	for (int i = 0; i < indices.size(); i ++) {
		updatePrio(indices[i], prios[i]);

		if (prios[i] > defaultMaxPrio) {
			defaultMaxPrio = prios[i];
		}
	}
}







