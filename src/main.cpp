#include "cnf/cnf_parser.h"
#include "solver/approximate_counter.h"
#include "utils/timer.h"
#include "cuda_interface.h"
#include <iostream>
#include <cstring>
#include <cmath>

using namespace std;
using namespace sharpsat;

// configuration
struct AppConfig {
    string input_file;
    double epsilon = 0.8;
    double delta = 0.2;
    bool use_ml = false;
    bool use_cuda = true;
    uint32_t seed = 42;
    double timeout_seconds = 60.0;
    uint32_t num_trials = 0;        // calculate from epsilon/delta
    bool verbose = false;
    bool show_gpu_info = false;
    uint32_t num_threads = 0;     // 0 = auto-detect
    
    // user can only specify one of epsilon/delta or trials
    bool epsilon_specified = false;
    bool delta_specified = false;
    bool trials_specified = false;
};

// calculate number of trials from epsilon and delta: iterations = 3 × ln(3/delta) / epsilon²
uint32_t calculate_trials(double epsilon, double delta) {
    double iterations = 3.0 * log(3.0 / delta) / (epsilon * epsilon);
    return static_cast<uint32_t>(ceil(iterations));
}

// calculate epsilon from trials (with standard delta=0.2): iterations = 3 × ln(3/δ) / ε²
double calculate_epsilon(uint32_t trials, double delta = 0.2) {
    double epsilon_sq = 3.0 * log(3.0 / delta) / trials;
    return sqrt(epsilon_sq);
}

void print_usage(const char* prog_name) {
    cout << "Usage: " << prog_name << " <cnf_file> [options]" << endl;
    cout << "Options:" << endl;
    cout << "  --epsilon <float>   Approximation factor (default: 0.8) - Note: Cannot be used with --trials" << endl;
    cout << "  --delta <float>     Confidence parameter (default: 0.2) - Note: Cannot be used with --trials" << endl;
    cout << "  --trials <int>      Number of trials to run (default: calculated from epsilon/delta) - Note: Cannot be used with --epsilon/--delta" << endl;
    cout << "  --timeout <float>   Timeout in seconds per SAT call (default: 60)" << endl;
    cout << "  --threads <int>     CPU threads for parallel trials (default: auto)" << endl;
    cout << "  --use-ml            Enable ML-enhanced hash generation" << endl;
    cout << "  --no-cuda           Disable CUDA acceleration" << endl;
    cout << "  --seed <int>        Random seed (default: 42)" << endl;
    cout << "  --verbose           Enable detailed logging" << endl;
    cout << "  --gpu-info          Show GPU information" << endl;
    cout << "  --help              Show this help message" << endl;
}

// parse command-line arguments
bool parse_args(int argc, char** argv, AppConfig& config) {
    if (argc < 2) {
        return false;
    }
    
    // check if --help is present anywhere in the arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--help") {
            return false;
        }
    }
    
    // first argument is the input CNF file
    config.input_file = argv[1];
    
    // parse options
    for (int i = 2; i < argc; i++) {
        string arg = argv[i];
        
        if (arg == "--help") {
            return false;
        } else if (arg == "--epsilon" && i + 1 < argc) {
            config.epsilon = stod(argv[++i]);
            config.epsilon_specified = true;
        } else if (arg == "--delta" && i + 1 < argc) {
            config.delta = stod(argv[++i]);
            config.delta_specified = true;
        } else if (arg == "--timeout" && i + 1 < argc) {
            config.timeout_seconds = stod(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            config.num_threads = stoul(argv[++i]);
        } else if (arg == "--trials" && i + 1 < argc) {
            config.num_trials = stoul(argv[++i]);
            config.trials_specified = true;
        } else if (arg == "--use-ml") {
            config.use_ml = true;
        } else if (arg == "--no-cuda") {
            config.use_cuda = false;
        } else if (arg == "--seed" && i + 1 < argc) {
            config.seed = stoul(argv[++i]);
        } else if (arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--gpu-info") {
            config.show_gpu_info = true;
        } else {
            cerr << "Unknown option: " << arg << endl;
            return false;
        }
    }
    
    // make sure that only one of trials or epsilon/delta is specified and calculate the other
    if (config.trials_specified && (config.epsilon_specified || config.delta_specified)) {
        cerr << "Error: Cannot specify both --trials and --epsilon/--delta" << endl;
        cerr << "Use --trials to directly set iterations, OR use --epsilon/--delta to calculate them" << endl;
        return false;
    }
    if (!config.trials_specified) {
        config.num_trials = calculate_trials(config.epsilon, config.delta);
    } else {
        config.delta = 0.2;
        config.epsilon = calculate_epsilon(config.num_trials, config.delta);
    }
    
    return true;
}

// main function - runs the program
int main(int argc, char** argv) {
    // parse command-line arguments
    AppConfig config;
    if (!parse_args(argc, argv, config)) {
        print_usage(argv[0]);
        return 1;
    }
    
    // print header
    cout << "GPU-Accelerated Approximate Model Counter" << endl;
    cout << "=========================================" << endl;
    
    // parse CNF file
    if (config.verbose) {
        cout << "Parsing CNF file: " << config.input_file << endl;
    }
    Timer parse_timer;
    auto cnf = CNFParser::parse_file(config.input_file);
    
    if (!cnf) {
        cerr << "  ERROR: Failed to parse CNF file" << endl;
        return 1;
    }
    
    if (config.verbose){
        cout << "  Parsed formula in " << parse_timer.elapsed_seconds() << " seconds" << endl;
        cout << "  Variables: " << cnf->num_variables() << endl;
        cout << "  Clauses: " << cnf->num_clauses() << endl;
        cout << "  Average clause size: " << cnf->avg_clause_size() << endl;
    }
    
    // configure counter
    CounterConfig counter_config;
    counter_config.epsilon = config.epsilon;
    counter_config.delta = config.delta;
    counter_config.seed = config.seed;
    counter_config.use_ml_hashes = config.use_ml;
    counter_config.use_cuda = config.use_cuda;
    counter_config.timeout_seconds = config.timeout_seconds;
    counter_config.num_trials = config.num_trials;
    counter_config.num_threads = config.num_threads;
    
    // show configuration
    cout << "Configuration:" << endl;
    cout << "  Epsilon: " << counter_config.epsilon << endl;
    cout << "  Delta: " << counter_config.delta << endl;
    cout << "  Timeout: " << counter_config.timeout_seconds << " seconds" << endl;
    cout << "  Trials: " << counter_config.num_trials << (config.trials_specified ? " (user-specified)" : " (calculated from epsilon/delta)") << endl;
    if (config.trials_specified) {
        cout << "  Note: Epsilon calculated from trials for delta=0.2" << endl;
    }
    
    if (config.use_ml) {
        cout << "  ML-enhanced hash generation: ENABLED" << endl;
    }

    // check CUDA availability
    if (config.use_cuda) {
        if (cuda::is_cuda_available()) {
            cout << "  CUDA: Enabled" << endl;            if (config.show_gpu_info) {
                cuda::print_gpu_info(0);
            }
        } else {
            cerr << "  WARNING: CUDA not available, falling back to CPU" << endl;
            config.use_cuda = false;
        }
    }
    
    // run approximate counting
    cout << endl << "Running approximate model counting on " << config.input_file << endl;
    ApproximateCounter counter(counter_config);
    CountResult result = counter.count(*cnf);
    
    // print results
    cout << endl << "Results:" << endl;
    if (result.successful) {
        // Display log10 count first — accurate even when the raw double overflows
        cout << "  Log10(Count):        " << result.log10_count << endl;
        cout << "  Order of Magnitude:  10^" << result.order_of_magnitude() << endl;
        cout << "  Approximate Model Count: " << result.count << endl;
        cout << "  Lower Bound (log10): " << result.log10_lower_bound << endl;
        cout << "  Upper Bound (log10): " << result.log10_upper_bound << endl;
        cout << "  Lower Bound: " << result.lower_bound << endl;
        cout << "  Upper Bound: " << result.upper_bound << endl;
        cout << "  Iterations: " << result.num_iterations << endl;
        cout << "  Time: " << result.time_seconds << " seconds" << endl;
    } else {
        cerr << "  ERROR: Counting failed" << endl;
        return 1;
    }
    
    return 0;
}
