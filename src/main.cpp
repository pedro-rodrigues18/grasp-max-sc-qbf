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
    bool file_exists = std::filesystem::exists(filename);
    
    std::ofstream file(filename, std::ios::app); // append
    if (!file_exists) {
        file << "Instance,Configuration,Value,Time_Seconds,Feasible\n";
    }

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
        GRASP grasp(alpha, 10000, 600, cm, sm); // 10 minutes time limit

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

#include <chrono>
#include <iomanip>

void runInstance(const std::string& instPath, const std::string& instName) {
    std::vector<std::tuple<std::string, GRASP::ConstructionMethod, GRASP::SearchMethod, double>> configs = {
        {"STANDARD+ALPHA", GRASP::STANDARD, GRASP::FIRST_IMPROVING, 0.3}
    };

    std::string baseName = instName;
    size_t lastindex = baseName.find_last_of(".");
    if (lastindex != std::string::npos) {
        baseName = baseName.substr(0, lastindex);
    }

    std::vector<std::future<ExperimentResult>> futures;
    for (auto& [cfgName, cm, sm, alpha] : configs) {
        futures.push_back(std::async(std::launch::async, [&, cfgName, cm, sm, alpha]() {
            return runSingleConfig(instPath, instName, cfgName, cm, sm, alpha);
        }));
    }

    std::vector<ExperimentResult> local_results;
    for (auto& f : futures) {
        local_results.push_back(f.get());
    }

    // Abrir em modo append para não sobrescrever
    std::ofstream log("logs/" + baseName + ".log", std::ios::app);

    // Adicionar timestamp para diferenciar execuções
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    log << "=== Execution at " << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S") << " ===\n";

    log << "Running instance: " << instName << "\n";
    for (auto& r : local_results) {
        log << r.config << " -> Value=" << r.value
            << " Time=" << r.time_seconds << "s"
            << " Feasible=" << (r.feasible ? "Yes" : "No") << "\n";
    }
    log << "\n";

    log.close();

    {
        std::lock_guard<std::mutex> lock(results_mutex);
        all_results.insert(all_results.end(), local_results.begin(), local_results.end());
    }
}


void runAllInstances(const std::vector<std::string>& instances) {
    // unsigned int num_threads = std::min(16u, std::max(1u, std::thread::hardware_concurrency()));
    unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
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
