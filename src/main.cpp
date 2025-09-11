#include<iostream>
#include"sc-qbf/sc_qbf.hpp"
#include "grasp/grasp.hpp"
#include <chrono>

int main() {
    SetCoverQBF scqbf("instances/ex-06.txt");

    // vector<int> solution = { 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 };

    vector<double> alphaValues = { 0.1, 0.3 };

    cout << "=== GRASP EXPERIMENTS FOR MAX-SC-QBF ===" << endl;
    cout << "Time limit per run: 30 minutes" << endl << endl;

    // 1. STANDARD: GRASP with parameter α1, first-improving and standard constructive heuristic
    cout << "1. STANDARD CONFIGURATION (alpha=0.1, first-improving, standard)" << endl;
    GRASP grasp1(alphaValues[0], 1000, 1800, GRASP::STANDARD, GRASP::FIRST_IMPROVING);

    auto start = chrono::high_resolution_clock::now();
    vector<int> solution1 = grasp1.run(scqbf);
    auto end = chrono::high_resolution_clock::now();
    auto duration1 = chrono::duration_cast<chrono::seconds>(end - start);

    double value1 = scqbf.evaluateSolution(solution1);
    bool feasible1 = scqbf.isFeasible(solution1);
    cout << "Result: Value = " << value1 << ", Feasible = " << (feasible1 ? "Yes" : "No")
        << ", Time = " << duration1.count() << "s" << endl << endl;

    // 2. STANDARD+ALPHA: STANDARD GRASP but with parameter α2
    cout << "2. STANDARD+ALPHA (alpha=0.3, first-improving, standard)" << endl;
    GRASP grasp2(alphaValues[1], 1000, 1800, GRASP::STANDARD, GRASP::FIRST_IMPROVING);

    start = chrono::high_resolution_clock::now();
    vector<int> solution2 = grasp2.run(scqbf);
    end = chrono::high_resolution_clock::now();
    auto duration2 = chrono::duration_cast<chrono::seconds>(end - start);

    double value2 = scqbf.evaluateSolution(solution2);
    bool feasible2 = scqbf.isFeasible(solution2);
    cout << "Result: Value = " << value2 << ", Feasible = " << (feasible2 ? "Yes" : "No")
        << ", Time = " << duration2.count() << "s" << endl << endl;

    // 3. STANDARD+BEST: STANDARD GRASP but with best-improving
    cout << "3. STANDARD+BEST (alpha=0.1, best-improving, standard)" << endl;
    GRASP grasp3(alphaValues[0], 1000, 1800, GRASP::STANDARD, GRASP::BEST_IMPROVING);

    start = chrono::high_resolution_clock::now();
    vector<int> solution3 = grasp3.run(scqbf);
    end = chrono::high_resolution_clock::now();
    auto duration3 = chrono::duration_cast<chrono::seconds>(end - start);

    double value3 = scqbf.evaluateSolution(solution3);
    bool feasible3 = scqbf.isFeasible(solution3);
    cout << "Result: Value = " << value3 << ", Feasible = " << (feasible3 ? "Yes" : "No")
        << ", Time = " << duration3.count() << "s" << endl << endl;

    // 4. STANDARD+HC1: STANDARD GRASP but with alternative construction method 1 (Random + Greedy)
    cout << "4. STANDARD+HC1 (alpha=0.1, first-improving, random+greedy)" << endl;
    GRASP grasp4(alphaValues[0], 1000, 1800, GRASP::RANDOM_PLUS_GREEDY, GRASP::FIRST_IMPROVING);

    start = chrono::high_resolution_clock::now();
    vector<int> solution4 = grasp4.run(scqbf);
    end = chrono::high_resolution_clock::now();
    auto duration4 = chrono::duration_cast<chrono::seconds>(end - start);

    double value4 = scqbf.evaluateSolution(solution4);
    bool feasible4 = scqbf.isFeasible(solution4);
    cout << "Result: Value = " << value4 << ", Feasible = " << (feasible4 ? "Yes" : "No")
        << ", Time = " << duration4.count() << "s" << endl << endl;

    // 5. STANDARD+HC2: STANDARD GRASP but with alternative construction method 2 (Sampled Greedy)
    cout << "5. STANDARD+HC2 (alpha=0.1, first-improving, sampled-greedy)" << endl;
    GRASP grasp5(alphaValues[0], 1000, 1800, GRASP::SAMPLED_GREEDY, GRASP::FIRST_IMPROVING);

    start = chrono::high_resolution_clock::now();
    vector<int> solution5 = grasp5.run(scqbf);
    end = chrono::high_resolution_clock::now();
    auto duration5 = chrono::duration_cast<chrono::seconds>(end - start);

    double value5 = scqbf.evaluateSolution(solution5);
    bool feasible5 = scqbf.isFeasible(solution5);
    cout << "Result: Value = " << value5 << ", Feasible = " << (feasible5 ? "Yes" : "No")
        << ", Time = " << duration5.count() << "s" << endl << endl;

    cout << "=== SUMMARY OF RESULTS ===" << endl;
    cout << "Configuration\t\tValue\t\tTime(s)\t\tFeasible" << endl;
    cout << "STANDARD\t\t" << value1 << "\t\t" << duration1.count() << "\t\t" << (feasible1 ? "Yes" : "No") << endl;
    cout << "STANDARD+ALPHA\t\t" << value2 << "\t\t" << duration2.count() << "\t\t" << (feasible2 ? "Yes" : "No") << endl;
    cout << "STANDARD+BEST\t\t" << value3 << "\t\t" << duration3.count() << "\t\t" << (feasible3 ? "Yes" : "No") << endl;
    cout << "STANDARD+HC1\t\t" << value4 << "\t\t" << duration4.count() << "\t\t" << (feasible4 ? "Yes" : "No") << endl;
    cout << "STANDARD+HC2\t\t" << value5 << "\t\t" << duration5.count() << "\t\t" << (feasible5 ? "Yes" : "No") << endl;

    return 0;
}