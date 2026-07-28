#pragma once

#include <cstddef>
#include <utility>
#include <vector>

/*
┌───────────────────────────────────────────────────┐
│   Train-Test Split                                │
└───────────────────────────────────────────────────┘
*/


// Needs careful examination 

inline size_t SplitCut(size_t N, double testFrac) {
    return (size_t)((double)N - testFrac * (double)N);
}

struct Split { size_t trainEnd, testStart, N; };

inline Split TrainTestIndex(size_t N, double testFrac, size_t embargo) {
    size_t cut = SplitCut(N, testFrac);
    Split s;
    s.N         = N;
    s.trainEnd  = (cut > embargo) ? cut - embargo : 0;
    s.testStart = cut;
    return s;
}

inline std::pair<std::vector<double>, std::vector<double>> TrainTestSplit(
    const std::vector<double>& input, double testFrac, int embargo = 0
) {
    if (embargo < 0) embargo = 0;
    Split s = TrainTestIndex(input.size(), testFrac, (size_t)embargo);
    return { std::vector<double>(input.begin(), input.begin() + s.trainEnd),
             std::vector<double>(input.begin() + s.testStart, input.end()) };
}

/*
std::pair<std::vector<double>, std::vector<double>> TrainTestSplit(
    std::vector<double> input, double testFrac, int embargo = 0
) {

    if (embargo < 0) embargo = 0;
    // determine the size of an input 
    int N = input.size(); 

    // e.g. N = 1,000; testFrac = 0.2; -> 1,000 - 0.2 * 1,000 = 800 
    // i.e. before 800 - Train, after 800 - Test (add embargo!)

    std::vector<double> trainSet {}; 
    std::vector<double> testSet {};

    int TestSize = (int)(N - testFrac * N);     // cut stays where it is

    // Train
    for (size_t i = 0; i < (size_t)TestSize; ++i) {
        trainSet.push_back(input[i]);
    }

    // Test — embargo pushes the START, so those rows land in neither set
    for (size_t j = (size_t)(TestSize + embargo); j < (size_t)N; ++j) {
        testSet.push_back(input[j]);
    }
    return {trainSet, testSet};
}
*/