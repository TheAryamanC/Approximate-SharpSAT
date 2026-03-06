#include "cnf/cnf_parser.h"
#include "solver/approximate_counter.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "cuda_interface.h"
#include <iostream>
#include <cstring>
#include <cmath>

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
    double timeout_seconds = 60.0;
    uint32_t num_trials = 0;  // 0 means calculate from epsilon/delta
    bool verbose = false;
    bool show_gpu_info = false;
    
    // Track what user specified
    bool epsilon_specified = false;
    bool delta_specified = false;
    bool trials_specified = false;
};

// Calculate number of trials from epsilon and delta using ApproxMC formula
uint32_t calculate_trials(double epsilon, double delta) {
    // Formula: iterations = 3 × ln(3/δ) / ε²
    double iterations = 3.0 * std::log(3.0 / delta) / (epsilon * epsilon);
    return static_cast<uint32_t>(std::ceil(iterations));
}

// Back-calculate epsilon from trials (assuming delta=0.2)
double calculate_epsilon(uint32_t trials, double delta = 0.2) {
    // From: trials = 3 × ln(3/δ) / ε²
    // Solve for ε: ε = sqrt(3 × ln(3/δ) / trials)
    double epsilon_sq = 3.0 * std::log(3.0 / delta) / trials;
    return std::sqrt(epsilon_sq);
}

// Explain usage instructions
void print_usage(const char* prog_name) {
    cout << "Usage: " << prog_name << " <cnf_file> [options]\n"
         << "\nOptions:\n"
         << "  --epsilon <float>   Approximation factor (default: 0.8)\n"
         << "  --delta <float>     Confidence parameter (default: 0.2)\n"
         << "                      Note: Cannot be used with --trials\n"
         << "  --trials <int>      Number of trials to run\n"
         << "                      If specified, epsilon is calculated for delta=0.2\n"
         << "                      Cannot be used with --epsilon/--delta\n"
         << "  --timeout <float>   Timeout in seconds per SAT call (default: 60.0)\n"
         << "  --use-ml            Enable ML-enhanced hash generation\n"
         << "  --no-cuda           Disable CUDA acceleration\n"
         << "  --seed <int>        Random seed (default: 42)\n"
         << "  --verbose           Enable detailed logging\n"
         << "  --gpu-info          Show GPU information\n"
         << "  --help              Show this help message\n"
         << "\nDefault behavior:\n"
         << "  If neither --trials nor --epsilon/--delta is specified:\n"
         << "  - Uses epsilon=0.8, delta=0.2\n"
         << "  - Calculates trials = 3×ln(3/δ)/ε² ≈ 13\n"
         << "\nExamples:\n"
         << "  " << prog_name << " formula.cnf --epsilon 0.5 --delta 0.1\n"
         << "  " << prog_name << " formula.cnf --trials 50\n";
}

// Parse command-line arguments
bool parse_args(int argc, char** argv, AppConfig& config) {
    if (argc < 2) {
        return false;
    }
    
    // Check for help flag first
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            return false;
        }
    }
    
    config.input_file = argv[1];
    
    for (int i = 2; i < argc; i++) {
        string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            return false;
        } else if (arg == "--epsilon" && i + 1 < argc) {
            config.epsilon = stod(argv[++i]);
            config.epsilon_specified = true;
        } else if (arg == "--delta" && i + 1 < argc) {
            config.delta = std::stod(argv[++i]);
            config.delta_specified = true;
        } else if (arg == "--timeout" && i + 1 < argc) {
            config.timeout_seconds = std::stod(argv[++i]);
        } else if (arg == "--trials" && i + 1 < argc) {
            config.num_trials = std::stoul(argv[++i]);
            config.trials_specified = true;
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
    
    // Validate mutually exclusive options
    if (config.trials_specified && (config.epsilon_specified || config.delta_specified)) {
        std::cerr << "Error: Cannot specify both --trials and --epsilon/--delta\n";
        std::cerr << "Use --trials to directly set iterations, OR use --epsilon/--delta to calculate them.\n";
        return false;
    }
    
    // Calculate trials from epsilon/delta if not specified
    if (!config.trials_specified) {
        config.num_trials = calculate_trials(config.epsilon, config.delta);
    } else {
        // Back-calculate epsilon from trials (with delta=0.2)
        config.delta = 0.2;
        config.epsilon = calculate_epsilon(config.num_trials, config.delta);
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
    counter_config.timeout_seconds = config.timeout_seconds;
    counter_config.num_trials = config.num_trials;
    
    // Log parameter configuration
    LOG_INFO("Configuration:");
    LOG_INFO("  Epsilon: ", counter_config.epsilon);
    LOG_INFO("  Delta: ", counter_config.delta);
    LOG_INFO("  Trials: ", counter_config.num_trials, 
             config.trials_specified ? " (user-specified)" : " (calculated from epsilon/delta)");
    if (config.trials_specified) {
        LOG_INFO("  Note: Epsilon back-calculated from trials for delta=0.2");
    }
    
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
