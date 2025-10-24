#ifndef GRASP_HPP
#define GRASP_HPP

#include "../sc-qbf/sc_qbf.hpp"
#include <vector>
#include <random>
#include <set>
#include <chrono>

using namespace std;

class GRASP {
public:
    enum ConstructionMethod {
        STANDARD,
        RANDOM_PLUS_GREEDY,
        SAMPLED_GREEDY
    };

    enum SearchMethod {
        FIRST_IMPROVING,
        BEST_IMPROVING
    };

private:
    double alpha; // RCL parametters [0,1]
    int maxIterations;
    int timeLimit; // Time limit in seconds
    ConstructionMethod constructionMethod;
    SearchMethod searchMethod;
    mutable mt19937 rng; // Random number generator

public:
    GRASP();
    GRASP(double alpha, int maxIter, int timeLimit, ConstructionMethod cm = STANDARD, SearchMethod sm = FIRST_IMPROVING);

    vector<int> run(const SetCoverQBF& scqbf);

    // Setters
    void setAlpha(double a) { alpha = a; }
    void setMaxIterations(int maxIter) { maxIterations = maxIter; }
    void setTimeLimit(int timeLimit) { this->timeLimit = timeLimit; }
    void setConstructionMethod(ConstructionMethod cm) { constructionMethod = cm; }
    void setSearchMethod(SearchMethod sm) { searchMethod = sm; }

    // Getters
    double getAlpha() const { return alpha; }
    int getMaxIterations() const { return maxIterations; }
    int getTimeLimit() const { return timeLimit; }
    ConstructionMethod getConstructionMethod() const { return constructionMethod; }
    SearchMethod getSearchMethod() const { return searchMethod; }

private:
    vector<int> constructSolution(const SetCoverQBF& scqbf, const chrono::high_resolution_clock::time_point& startTime);
    vector<int> constructStandard(const SetCoverQBF& scqbf, const chrono::high_resolution_clock::time_point& startTime);
    vector<int> constructRandomPlusGreedy(const SetCoverQBF& scqbf);
    vector<int> constructSampledGreedy(const SetCoverQBF& scqbf);
    vector<int> repairSolution(const SetCoverQBF& scqbf, vector<int> solution) const;

    double calculateBenefit(const SetCoverQBF& scqbf, const vector<int>& currentSolution,
        int candidate, const set<int>& uncoveredElements) const;
    vector<int> buildRCL(const vector<pair<double, int>>& candidateBenefits) const;
    void updateUncoveredElements(const SetCoverQBF& scqbf, int selectedSet,
        set<int>& uncoveredElements) const;
    vector<int> localSearch(const SetCoverQBF& scqbf, vector<int> solution, const chrono::high_resolution_clock::time_point& startTime) const;
    vector<int> localSearchFirstImproving(const SetCoverQBF& scqbf, vector<int> solution, const chrono::high_resolution_clock::time_point& startTime) const;
    vector<int> localSearchBestImproving(const SetCoverQBF& scqbf, vector<int> solution) const;
};

#endif