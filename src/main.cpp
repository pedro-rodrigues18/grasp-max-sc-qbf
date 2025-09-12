#include <iostream>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <thread>
#include <mutex>
#include <vector>
#include <future>
#include <iomanip>
#include <sstream>
#include <map>
#include "sc-qbf/sc_qbf.hpp"
#include "grasp/grasp.hpp"

std::mutex results_mutex;

struct ExperimentResult {
    std::string instance;
    std::string config;
    double value;
    int time_seconds;
    bool feasible;
};

std::vector<ExperimentResult> all_results;

void writeResults(const std::string& filename) {
    std::ofstream file(filename);
    file << "Instance,Configuration,Value,Time_Seconds,Feasible\n";
    for (const auto& r : all_results) {
        file << r.instance << "," << r.config << ","
             << std::fixed << std::setprecision(2) << r.value << ","
             << r.time_seconds << ","
             << (r.feasible ? "Yes" : "No") << "\n";
    }
}

ExperimentResult runSingleConfig(const std::string& instPath, const std::string& instName,
                                 const std::string& cfgName, GRASP::ConstructionMethod cm,
                                 GRASP::SearchMethod sm, double alpha) {
    ExperimentResult r{instName, cfgName, -1, -1, false};
    try {
        SetCoverQBF scqbf(instPath);
        GRASP grasp(alpha, 10000, 1800, cm, sm);

        auto start = std::chrono::high_resolution_clock::now();
        auto sol = grasp.run(scqbf);
        auto end = std::chrono::high_resolution_clock::now();

        r.value = scqbf.evaluateSolution(sol);
        r.feasible = scqbf.isFeasible(sol);
        r.time_seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    } catch (const std::exception& e) {
        std::cerr << "Error in " << instName << ": " << e.what() << std::endl;
    }
    return r;
}

void runInstance(const std::string& instPath, const std::string& instName) {
    std::vector<std::tuple<std::string, GRASP::ConstructionMethod, GRASP::SearchMethod, double>> configs = {
        {"STANDARD", GRASP::STANDARD, GRASP::FIRST_IMPROVING, 0.1},
        {"STANDARD+ALPHA", GRASP::STANDARD, GRASP::FIRST_IMPROVING, 0.3},
        {"STANDARD+BEST", GRASP::STANDARD, GRASP::BEST_IMPROVING, 0.1},
        {"STANDARD+HC1", GRASP::RANDOM_PLUS_GREEDY, GRASP::FIRST_IMPROVING, 0.1},
        {"STANDARD+HC2", GRASP::SAMPLED_GREEDY, GRASP::FIRST_IMPROVING, 0.1}
    };

    std::string baseName = instName;
    size_t lastindex = baseName.find_last_of(".");
    baseName = baseName.substr(0, lastindex);
    std::ofstream log("logs/" + baseName + ".log");
    log << "Running instance: " << instName << "\n";

    std::vector<ExperimentResult> local_results;
    for (auto& [cfgName, cm, sm, alpha] : configs) {
        auto r = runSingleConfig(instPath, instName, cfgName, cm, sm, alpha);
        local_results.push_back(r);
        log << cfgName << " -> Value=" << r.value
            << " Time=" << r.time_seconds << "s"
            << " Feasible=" << (r.feasible ? "Yes" : "No") << "\n";
    }
    log.close();

    std::lock_guard<std::mutex> lock(results_mutex);
    all_results.insert(all_results.end(), local_results.begin(), local_results.end());
}

void runAllInstances(const std::vector<std::string>& instances) {
    unsigned int num_threads = std::min(16u, std::max(1u, std::thread::hardware_concurrency()));
    std::cout << "Using " << num_threads << " threads.\n";

    std::atomic<size_t> next(0);

    std::vector<std::future<void>> futures;
    for (unsigned int t = 0; t < num_threads; t++) {
        futures.push_back(std::async(std::launch::async, [&]() {
            while (true) {
                size_t idx = next.fetch_add(1);
                if (idx >= instances.size()) break;
                runInstance("instances/" + instances[idx], instances[idx]);
            }
        }));
    }
    for (auto& f : futures) f.wait();
}

std::vector<std::string> setupInstances(const std::string& path) {
    std::vector<std::string> insts;
    for (auto& e : std::filesystem::directory_iterator(path))
        if (e.is_regular_file()) insts.push_back(e.path().filename().string());
    std::sort(insts.begin(), insts.end());
    return insts;
}

int main() {
    std::filesystem::create_directory("logs");
    std::string path = "instances/";

    auto instances = setupInstances(path);
    if (instances.empty()) {
        std::cerr << "No instance found in " << path << std::endl;
        return 1;
    }

    std::cout << "Running " << instances.size() << " instances...\n";
    runAllInstances(instances);

    writeResults("grasp_results.csv");
    std::cout << "Results saved in grasp_results.csv\n";
    return 0;
}
