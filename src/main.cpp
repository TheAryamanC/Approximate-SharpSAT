#include "cnf/cnf_parser.h"
#include "solver/approximate_counter.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "cuda_interface.h"
#include <iostream>
#include <cstring>

using namespace std;
using namespace sharpsat;

// Application configuration structure
struct AppConfig {
    string input_file;
    double epsilon = 0.8;
    double delta = 0.2;
    bool use_ml = false;
    bool use_cuda = true;
    uint32_t seed = 42;
    bool verbose = false;
    bool show_gpu_info = false;
};

// Explain usage instructions
void print_usage(const char* prog_name) {
    cout << "Usage: " << prog_name << " <cnf_file> [options]\n"
         << "\nOptions:\n"
         << "  --epsilon <float>   Approximation factor (default: 0.8)\n"
         << "  --delta <float>     Confidence parameter (default: 0.2)\n"
         << "  --use-ml            Enable ML-enhanced hash generation\n"
         << "  --no-cuda           Disable CUDA acceleration\n"
         << "  --seed <int>        Random seed (default: 42)\n"
         << "  --verbose           Enable detailed logging\n"
         << "  --gpu-info          Show GPU information\n"
         << "  --help              Show this help message\n"
         << "\nExample:\n"
         << "  " << prog_name << " formula.cnf --epsilon 0.8 --delta 0.2 --use-ml\n";
}

// Parse command-line arguments
bool parse_args(int argc, char** argv, AppConfig& config) {
    if (argc < 2) {
        return false;
    }
    
    config.input_file = argv[1];
    
    for (int i = 2; i < argc; i++) {
        string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            return false;
        } else if (arg == "--epsilon" && i + 1 < argc) {
            config.epsilon = stod(argv[++i]);
        } else if (arg == "--delta" && i + 1 < argc) {
            config.delta = std::stod(argv[++i]);
        } else if (arg == "--use-ml") {
            config.use_ml = true;
        } else if (arg == "--no-cuda") {
            config.use_cuda = false;
        } else if (arg == "--seed" && i + 1 < argc) {
            config.seed = std::stoul(argv[++i]);
        } else if (arg == "--verbose" || arg == "-v") {
            config.verbose = true;
        } else if (arg == "--gpu-info") {
            config.show_gpu_info = true;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return false;
        }
    }
    
    return true;
}

// Main entry point
int main(int argc, char** argv) {
    AppConfig config;
    
    if (!parse_args(argc, argv, config)) {
        print_usage(argv[0]);
        return 1;
    }
    
    // set logging level
    if (config.verbose) {
        Logger::instance().set_verbose(true);
    }
    
    LOG_INFO("SharpSAT: GPU-Accelerated Approximate Model Counter");
    LOG_INFO("===================================================");
    
    // check CUDA availability
    if (config.use_cuda) {
        if (cuda::is_cuda_available()) {
            LOG_INFO("CUDA is available");
            if (config.show_gpu_info) {
                cuda::print_gpu_info(0);
            }
        } else {
            LOG_WARNING("CUDA not available, falling back to CPU");
            config.use_cuda = false;
        }
    }
    
    // parse CNF file
    LOG_INFO("Parsing CNF file: ", config.input_file);
    Timer parse_timer;
    auto cnf = CNFParser::parse_file(config.input_file);
    
    if (!cnf) {
        LOG_ERROR("Failed to parse CNF file");
        return 1;
    }
    
    LOG_INFO("Parsed formula in ", parse_timer.elapsed_seconds(), " seconds");
    LOG_INFO("Variables: ", cnf->num_variables());
    LOG_INFO("Clauses: ", cnf->num_clauses());
    LOG_INFO("Average clause size: ", cnf->avg_clause_size());
    
    // configure counter
    CounterConfig counter_config;
    counter_config.epsilon = config.epsilon;
    counter_config.delta = config.delta;
    counter_config.seed = config.seed;
    counter_config.use_ml_hashes = config.use_ml;
    counter_config.use_cuda = config.use_cuda;
    
    if (config.use_ml) {
        LOG_INFO("ML-enhanced hash generation: ENABLED");
    }
    
    // run approximate counting
    LOG_INFO("\nStarting approximate model counting...");
    LOG_INFO("---------------------------------------");
    
    ApproximateCounter counter(counter_config);
    CountResult result = counter.count(*cnf);
    
    // print results
    LOG_INFO("\n=== Results ===");
    if (result.successful) {
        std::cout << "\nApproximate Model Count: " << result.count << std::endl;
        std::cout << "Lower Bound: " << result.lower_bound << std::endl;
        std::cout << "Upper Bound: " << result.upper_bound << std::endl;
        std::cout << "Iterations: " << result.num_iterations << std::endl;
        std::cout << "Time: " << result.time_seconds << " seconds" << std::endl;
        
        LOG_INFO("\nCount: ", result.count);
        LOG_INFO("Bounds: [", result.lower_bound, ", ", result.upper_bound, "]");
        LOG_INFO("Iterations: ", result.num_iterations);
        LOG_INFO("Time: ", result.time_seconds, " seconds");
    } else {
        LOG_ERROR("Counting failed");
        return 1;
    }
    
    // print timing statistics
    if (config.verbose) {
        LOG_INFO("\n=== Timing Statistics ===");
        TimerRegistry::instance().print_all();
    }
    
    return 0;
}
