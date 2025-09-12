#include "grasp.hpp"
#include <algorithm>
#include <random>
#include <chrono>
#include <set>
#include <iostream>
#include <climits>

GRASP::GRASP() : alpha(0.1), maxIterations(1000), timeLimit(1800), // 30 minutes
constructionMethod(STANDARD), searchMethod(FIRST_IMPROVING) {
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    rng.seed(seed);
}

GRASP::GRASP(double alpha, int maxIter, int timeLimit, ConstructionMethod cm, SearchMethod sm)
    : alpha(alpha), maxIterations(maxIter), timeLimit(timeLimit),
    constructionMethod(cm), searchMethod(sm) {
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    rng.seed(seed);
}

vector<int> GRASP::run(const SetCoverQBF& scqbf) {
    // cout << "Iniciando GRASP para MAX-SC-QBF..." << endl;
    cout << "Starting GRASP for MAX-SC-QBF..." << endl;
    cout << "Parameters: alpha=" << alpha << ", maxIter=" << maxIterations << endl;

    vector<int> bestSolution;
    double bestValue = -1e9;

    auto startTime = chrono::high_resolution_clock::now();

    for (int iter = 0; iter < maxIterations; iter++) {
        auto currentTime = chrono::high_resolution_clock::now();
        auto elapsed = chrono::duration_cast<chrono::seconds>(currentTime - startTime).count();
        if (elapsed >= timeLimit) {
            cout << "Time limit reached!" << endl;
            break;
        }

        // Construction Phase
        vector<int> solution = constructSolution(scqbf);

        // Local Search Phase
        solution = localSearch(scqbf, solution);

        // Evaluate solution
        double value = scqbf.evaluateSolution(solution);

        // Update best solution
        if (value > bestValue) {
            bestValue = value;
            bestSolution = solution;
            cout << "New best solution found at iteration " << (iter + 1)
                << " with value: " << bestValue << endl;
        }

        if ((iter + 1) % 100 == 0) {
            cout << "Iteration " << (iter + 1) << " - Best value: " << bestValue << endl;
        }
    }

    cout << "GRASP finished. Best value found: " << bestValue << endl;

    return bestSolution;
}

vector<int> GRASP::constructSolution(const SetCoverQBF& scqbf) {
    switch (constructionMethod) {
    case RANDOM_PLUS_GREEDY:
        return constructRandomPlusGreedy(scqbf);
    case SAMPLED_GREEDY:
        return constructSampledGreedy(scqbf);
    default:
        return constructStandard(scqbf);
    }
}

vector<int> GRASP::constructStandard(const SetCoverQBF& scqbf) {
    int n = scqbf.getNumSets();
    vector<int> solution(n, 0);
    set<int> uncoveredElements = scqbf.getUniverse();
    vector<bool> candidateSet(n, true);

    while (!uncoveredElements.empty()) {
        vector<pair<double, int>> candidateBenefits;

        for (int i = 0; i < n; i++) {
            if (!candidateSet[i]) continue;

            double benefit = calculateBenefit(scqbf, solution, i, uncoveredElements);
            candidateBenefits.push_back({ benefit, i });
        }

        if (candidateBenefits.empty()) break;

        // Sort candidates by benefit (highest first)
        sort(candidateBenefits.begin(), candidateBenefits.end(), greater<pair<double, int>>());

        vector<int> rcl = buildRCL(candidateBenefits);

        // Select random element from RCL
        uniform_int_distribution<int> dist(0, rcl.size() - 1);
        int selectedIndex = rcl[dist(rng)];

        // Add selected set to solution
        solution[selectedIndex] = 1;
        candidateSet[selectedIndex] = false;

        // Update uncovered elements
        updateUncoveredElements(scqbf, selectedIndex, uncoveredElements);
    }

    return solution;
}

vector<int> GRASP::constructRandomPlusGreedy(const SetCoverQBF& scqbf) {
    int n = scqbf.getNumSets();
    vector<int> solution(n, 0);
    set<int> uncoveredElements = scqbf.getUniverse();
    vector<bool> candidateSet(n, true);

    // Random phase: select some sets randomly
    uniform_real_distribution<double> prob(0.0, 1.0);
    for (int i = 0; i < n; i++) {
        if (prob(rng) < 0.3) { // 30% of chance to select randomly
            solution[i] = 1;
            candidateSet[i] = false;
            updateUncoveredElements(scqbf, i, uncoveredElements);
        }
    }

    // Greedy phase: complete solution with greedy choices
    while (!uncoveredElements.empty()) {
        int bestCandidate = -1;
        double bestBenefit = -1e9;

        for (int i = 0; i < n; i++) {
            if (!candidateSet[i]) continue;

            double benefit = calculateBenefit(scqbf, solution, i, uncoveredElements);
            if (benefit > bestBenefit) {
                bestBenefit = benefit;
                bestCandidate = i;
            }
        }

        if (bestCandidate == -1) break;

        solution[bestCandidate] = 1;
        candidateSet[bestCandidate] = false;
        updateUncoveredElements(scqbf, bestCandidate, uncoveredElements);
    }

    return solution;
}

vector<int> GRASP::constructSampledGreedy(const SetCoverQBF& scqbf) {
    int n = scqbf.getNumSets();
    vector<int> solution(n, 0);
    set<int> uncoveredElements = scqbf.getUniverse();
    vector<bool> candidateSet(n, true);

    int sampleSize = max(1, n / 4); // 25% of candidates

    while (!uncoveredElements.empty()) {
        vector<int> sampledCandidates;
        for (int i = 0; i < n; i++) {
            if (candidateSet[i]) {
                sampledCandidates.push_back(i);
            }
        }

        if (sampledCandidates.empty()) break;

        // Select random sample
        shuffle(sampledCandidates.begin(), sampledCandidates.end(), rng);
        int actualSampleSize = min(sampleSize, (int)sampledCandidates.size());
        sampledCandidates.resize(actualSampleSize);

        // Calculate benefits for sampled candidates
        vector<pair<double, int>> candidateBenefits;
        for (int candidate : sampledCandidates) {
            double benefit = calculateBenefit(scqbf, solution, candidate, uncoveredElements);
            candidateBenefits.push_back({ benefit, candidate });
        }

        // Sort and build RCL
        sort(candidateBenefits.begin(), candidateBenefits.end(), greater<pair<double, int>>());
        vector<int> rcl = buildRCL(candidateBenefits);

        // Select random element from RCL
        uniform_int_distribution<int> dist(0, rcl.size() - 1);
        int selectedIndex = rcl[dist(rng)];

        solution[selectedIndex] = 1;
        candidateSet[selectedIndex] = false;
        updateUncoveredElements(scqbf, selectedIndex, uncoveredElements);
    }

    return solution;
}

double GRASP::calculateBenefit(const SetCoverQBF& scqbf, const vector<int>& currentSolution,
    int candidate, const set<int>& uncoveredElements) const {

    double benefit = 0.0;

    const vector<int>& candidateSet = scqbf.getSet(candidate);
    int newElementsCovered = 0;
    for (int element : candidateSet) {
        if (uncoveredElements.count(element) > 0) {
            newElementsCovered++;
        }
    }

    // Weight high for coverage (priority on feasibility)
    benefit += newElementsCovered * 100.0;

    // Adding objective function contribution
    benefit += scqbf.getLinearCoeff(candidate); // Linear term

    // Quadratic terms with already selected sets
    for (int i = 0; i < static_cast<int>(currentSolution.size()); i++) {
        if (currentSolution[i] == 1) {
            benefit += scqbf.getQuadraticCoeff(min(i, candidate), max(i, candidate));
        }
    }

    return benefit;
}

vector<int> GRASP::buildRCL(const vector<pair<double, int>>& candidateBenefits) const {
    if (candidateBenefits.empty()) return {};

    double maxBenefit = candidateBenefits[0].first;
    double minBenefit = candidateBenefits.back().first;
    double threshold = minBenefit + alpha * (maxBenefit - minBenefit);

    vector<int> rcl;
    for (const auto& candidate : candidateBenefits) {
        if (candidate.first >= threshold) {
            rcl.push_back(candidate.second);
        }
    }

    return rcl;
}

void GRASP::updateUncoveredElements(const SetCoverQBF& scqbf, int selectedSet,
    set<int>& uncoveredElements) const {
    const vector<int>& setElements = scqbf.getSet(selectedSet);
    for (int element : setElements) {
        uncoveredElements.erase(element);
    }
}

vector<int> GRASP::localSearch(const SetCoverQBF& scqbf, vector<int> solution) const {
    switch (searchMethod) {
    case BEST_IMPROVING:
        return localSearchBestImproving(scqbf, solution);
    default:
        return localSearchFirstImproving(scqbf, solution);
    }
}

vector<int> GRASP::localSearchFirstImproving(const SetCoverQBF& scqbf, vector<int> solution) const {
    bool improved = true;
    double currentValue = scqbf.evaluateSolution(solution);

    while (improved) {
        improved = false;

        // Operator 1: Flip (toggle 0->1 or 1->0)
        for (int i = 0; i < static_cast<int>(solution.size()); i++) {
            vector<int> neighbor = solution;
            neighbor[i] = 1 - neighbor[i];

            if (scqbf.isFeasible(neighbor)) {
                double neighborValue = scqbf.evaluateSolution(neighbor);
                if (neighborValue > currentValue) {
                    solution = neighbor;
                    currentValue = neighborValue;
                    improved = true;
                    break;
                }
            }
        }

        if (improved) continue;

        // Operator 2: Swap (swap states of two sets)
        for (int i = 0; i < static_cast<int>(solution.size()) && !improved; i++) {
            for (int j = i + 1; j < static_cast<int>(solution.size()); j++) {
                if (solution[i] != solution[j]) {
                    vector<int> neighbor = solution;
                    swap(neighbor[i], neighbor[j]);

                    if (scqbf.isFeasible(neighbor)) {
                        double neighborValue = scqbf.evaluateSolution(neighbor);
                        if (neighborValue > currentValue) {
                            solution = neighbor;
                            currentValue = neighborValue;
                            improved = true;
                            break;
                        }
                    }
                }
            }
        }
    }

    return solution;
}

vector<int> GRASP::localSearchBestImproving(const SetCoverQBF& scqbf, vector<int> solution) const {
    bool improved = true;
    double currentValue = scqbf.evaluateSolution(solution);

    while (improved) {
        improved = false;
        vector<int> bestNeighbor = solution;
        double bestValue = currentValue;

        // Operator 1: Flip
        for (int i = 0; i < static_cast<int>(solution.size()); i++) {
            vector<int> neighbor = solution;
            neighbor[i] = 1 - neighbor[i];

            if (scqbf.isFeasible(neighbor)) {
                double neighborValue = scqbf.evaluateSolution(neighbor);
                if (neighborValue > bestValue) {
                    bestNeighbor = neighbor;
                    bestValue = neighborValue;
                    improved = true;
                }
            }
        }

        // Operator 2: Swap
        for (int i = 0; i < static_cast<int>(solution.size()); i++) {
            for (int j = i + 1; j < static_cast<int>(solution.size()); j++) {
                if (solution[i] != solution[j]) {
                    vector<int> neighbor = solution;
                    swap(neighbor[i], neighbor[j]);

                    if (scqbf.isFeasible(neighbor)) {
                        double neighborValue = scqbf.evaluateSolution(neighbor);
                        if (neighborValue > bestValue) {
                            bestNeighbor = neighbor;
                            bestValue = neighborValue;
                            improved = true;
                        }
                    }
                }
            }
        }

        if (improved) {
            solution = bestNeighbor;
            currentValue = bestValue;
        }
    }

    return solution;
}