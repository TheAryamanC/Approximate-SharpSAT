#include "xor/ml_hash_interface.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>

using namespace std;
namespace sharpsat {

// constructors
MLHashInterface::MLHashInterface() 
    : model_available_(false), fallback_generator_(make_unique<XorHashGenerator>()),
      rng_(42), uniform_dist_(0.0, 1.0), bool_dist_(0, 1), noise_dist_(0.0, 0.05),
      python_stdin_(nullptr), python_stdout_(nullptr), python_stderr_(nullptr), python_pid_(-1) {

}

MLHashInterface::MLHashInterface(const string& model_path)
    : model_path_(model_path), model_available_(false),
      fallback_generator_(make_unique<XorHashGenerator>()),
      rng_(42), uniform_dist_(0.0, 1.0), bool_dist_(0, 1), noise_dist_(0.0, 0.05),
      python_stdin_(nullptr), python_stdout_(nullptr), python_stderr_(nullptr), python_pid_(-1) {
    initialize(model_path);
}

// destructor
MLHashInterface::~MLHashInterface() {
    stop_ml_server();
}

// initialize ML model
bool MLHashInterface::initialize(const string& model_path) {
    model_path_ = model_path;
    
    // check if model file exists
    ifstream model_file(model_path);
    if (!model_file.good()) {
        cerr << "Warning: ML model file not found at " << model_path << endl;
        cerr << "         Using heuristic importance-weighted hash generation instead" << endl;
        model_available_ = false;
        return false;
    }
    
    // start persistent Python ML server (only open pipe once)
    if (start_ml_server()) {
        model_available_ = true;
        return true;
    } else {
        model_available_ = false;
        return false;
    }
}

// start persistent Python ML server
bool MLHashInterface::start_ml_server() {
    int stdin_pipe[2];
    int stdout_pipe[2];
    int stderr_pipe[2];
    
    // create pipes for communication with Python process
    if (pipe(stdin_pipe) < 0 || pipe(stdout_pipe) < 0 || pipe(stderr_pipe) < 0) {
        cerr << "ERROR: Failed to create pipes for ML server" << endl;
        return false;
    }
    
    // fork process to run Python ML server
    python_pid_ = fork();
    if (python_pid_ < 0) {
        cerr << "ERROR: Failed to fork ML server process" << endl;
        close(stdin_pipe[0]); close(stdin_pipe[1]);
        close(stdout_pipe[0]); close(stdout_pipe[1]);
        close(stderr_pipe[0]); close(stderr_pipe[1]);
        return false;
    }
    if (python_pid_ == 0) {
        // child process: run Python ML server
        close(stdin_pipe[1]);   // Close write end of stdin
        close(stdout_pipe[0]);  // Close read end of stdout
        close(stderr_pipe[0]);  // Close read end of stderr
        
        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        
        close(stdin_pipe[0]);
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);
        
        execl("/usr/bin/python3", "python3", "src/ml_model/ml_server.py", nullptr);
        _exit(1);  // If exec fails
    }
    
    // parent process: keep pipes open
    close(stdin_pipe[0]);   // Close read end of stdin
    close(stdout_pipe[1]);  // Close write end of stdout
    close(stderr_pipe[1]);  // Close write end of stderr
    
    python_stdin_ = fdopen(stdin_pipe[1], "w");
    python_stdout_ = fdopen(stdout_pipe[0], "r");
    python_stderr_ = fdopen(stderr_pipe[0], "r");
    
    if (!python_stdin_ || !python_stdout_) {
        cerr << "ERROR: Failed to open file streams for ML server" << endl;
        stop_ml_server();
        return false;
    }
    
    // wait for READY signal from Python
    char buffer[128];
    if (fgets(buffer, sizeof(buffer), python_stdout_)) {
        string ready_msg(buffer);
        if (ready_msg.find("READY") != string::npos) {
            return true;
        }
    }
    
    cerr << "ERROR: ML server failed to send READY signal" << endl;
    stop_ml_server();
    return false;
}

// stop persistent Python ML server
void MLHashInterface::stop_ml_server() {
    if (python_stdin_) {
        fprintf(python_stdin_, "QUIT\n");
        fflush(python_stdin_);
        fclose(python_stdin_);
        python_stdin_ = nullptr;
    }
    
    if (python_stdout_) {
        fclose(python_stdout_);
        python_stdout_ = nullptr;
    }
    
    if (python_stderr_) {
        fclose(python_stderr_);
        python_stderr_ = nullptr;
    }
    
    if (python_pid_ > 0) {
        int status;
        waitpid(python_pid_, &status, WNOHANG);
        
        // give process time to exit gracefully
        usleep(10000);  // 10ms
        
        // force kill if still running
        if (kill(python_pid_, 0) == 0) {
            kill(python_pid_, SIGTERM);
            usleep(10000);
            if (kill(python_pid_, 0) == 0) {
                kill(python_pid_, SIGKILL);
            }
        }
        
        waitpid(python_pid_, &status, 0);
        python_pid_ = -1;
    }
}

// extract features from CNF
CNFFeatures MLHashInterface::extract_features(const CNF& cnf) {
    CNFFeatures features;
    
    // basic features
    features.num_variables = cnf.num_variables();
    features.num_clauses = cnf.num_clauses();
    features.avg_clause_size = cnf.avg_clause_size();
    features.variable_clause_ratio = static_cast<double>(features.num_variables) / max(1u, features.num_clauses);
    
    // get variable occurrences
    const auto& occurrences = cnf.get_variable_occurrences();
    features.variable_occurrences.clear();
    features.variable_balance.clear();
    
    // compute max and min occurrences for normalization
    if (!occurrences.empty()) {
        features.max_occurrence = 0.0;
        features.min_occurrence = static_cast<double>(features.num_clauses);
        
        for (uint32_t var = 1; var <= features.num_variables; var++) {
            uint32_t occ = var < occurrences.size() ? occurrences[var] : 0;
            features.variable_occurrences.push_back(occ);
            
            if (occ > 0) {
                features.max_occurrence = max(features.max_occurrence, static_cast<double>(occ));
                features.min_occurrence = min(features.min_occurrence, static_cast<double>(occ));
            }
            
            // compute balance (ratio of positive to total occurrences)
            uint32_t pos = cnf.get_positive_occurrences(var);
            double balance = occ > 0 ? static_cast<double>(pos) / occ : 0.5;
            features.variable_balance.push_back(balance);
        }
    }
    
    return features;
}

// predict optimal hash configuration
HashConfig MLHashInterface::predict_hash_config(const CNF& cnf, uint32_t num_hashes) {
    HashConfig config;
    config.num_hashes = num_hashes;
    
    if (!model_available_) {
        // use default configuration if model is not available
        config.sparsity = XorHashGenerator::get_recommended_sparsity(cnf.num_variables());
        config.use_ml_predictor = false;
        return config;
    }
    
    // extract features and call ML predictor
    CNFFeatures features = extract_features(cnf);
    vector<double> predictions = call_python_predictor(features);
    
    if (!predictions.empty()) { // use ML predictions to set hash configuration
        config.sparsity = max(0.1, min(predictions[0], 0.9));
        config.use_ml_predictor = true;
    } else {
        config.sparsity = XorHashGenerator::get_recommended_sparsity(cnf.num_variables());
        config.use_ml_predictor = false;
    }
    
    return config;
}

// generate ML-enhanced XOR constraints
vector<XorConstraint> MLHashInterface::generate_ml_hashes(const CNF& cnf, uint32_t num_hashes) {
    // get variable importance scores (from ML model or heuristic fallback)
    vector<double> importance = predict_variable_importance(cnf);
    
    // get recommended sparsity for proper constraint density
    double sparsity = XorHashGenerator::get_recommended_sparsity(cnf.num_variables());
    
    // generate XOR constraints with biased sampling based on importance
    vector<XorConstraint> constraints;
    constraints.reserve(num_hashes);
    
    // for each hash, sample variables with probability = sparsity * importance_weight
    for (uint32_t i = 0; i < num_hashes; i++) {
        XorConstraint constraint;
        
        // sample variables based on importance-weighted probability
        for (uint32_t var = 1; var <= cnf.num_variables(); var++) {
            double importance_score = var - 1 < importance.size() ? importance[var - 1] : 0.5;
            // normalize importance to average 1.0, then multiply by base sparsity
            double probability = sparsity * (importance_score / 0.5);
            
            if (uniform_dist_(rng_) < probability) {
                constraint.variables.push_back(var);
            }
        }
        
        // ensure at least one variable is in the constraint
        if (constraint.variables.empty() && cnf.num_variables() > 0) {
            uniform_int_distribution<uint32_t> var_dist(1, cnf.num_variables());
            constraint.variables.push_back(var_dist(rng_));
        }
        
        constraint.rhs = bool_dist_(rng_) == 1;
        constraints.push_back(constraint);
    }
    
    return constraints;
}

// predict variable importance scores for XOR generation
vector<double> MLHashInterface::predict_variable_importance(const CNF& cnf) {
    uint32_t num_vars = cnf.num_variables();
    
    if (!model_available_ || !python_stdin_ || !python_stdout_) {
        // fallback to heuristics if ML not available
        const auto& occurrences = cnf.get_variable_occurrences();
        vector<double> importance(num_vars, 0.5);
        
        if (!occurrences.empty()) {
            uint32_t max_occ = 0;
            for (uint32_t var = 1; var <= num_vars; var++) {
                uint32_t occ = var < occurrences.size() ? occurrences[var] : 0;
                max_occ = max(max_occ, occ);
            }
            
            if (max_occ > 0) {
                for (uint32_t var = 1; var <= num_vars; var++) {
                    uint32_t occ = var < occurrences.size() ? occurrences[var] : 0;
                    importance[var - 1] = 0.2 + 0.6 * (static_cast<double>(occ) / max_occ);
                }
            }
        }
        
        return importance;
    }
    
    // use ML model via persistent Python server
    CNFFeatures features = extract_features(cnf);
    
    // add small Gaussian noise to continuous features for trial-to-trial variation (model is deterministic with default parameters)
    // otherwise we will get 0 variance between runs with the same formula
    double noisy_avg_clause = max(1.0, features.avg_clause_size + noise_dist_(rng_) * features.avg_clause_size);
    double noisy_ratio = max(0.01, features.variable_clause_ratio + noise_dist_(rng_) * features.variable_clause_ratio);
    
    // send request to Python server with original counts but noisy continuous features
    fprintf(
        python_stdin_, "%u %u %f %f", 
            features.num_variables,   // Keep exact
            features.num_clauses,     // Keep exact
            noisy_avg_clause,         // Add noise
            noisy_ratio               // Add noise
    );
    
    // send variable occurrences with small noise
    for (uint32_t occ : features.variable_occurrences) {
        double noisy_occ = max(0.0, occ + noise_dist_(rng_) * sqrt(static_cast<double>(occ + 1)));
        fprintf(python_stdin_, " %u", static_cast<uint32_t>(noisy_occ));
    }
    fprintf(python_stdin_, "\n");
    fflush(python_stdin_);
    
    // read importance scores from Python server
    vector<double> importance;
    importance.reserve(num_vars);
    
    char buffer[128];
    while (fgets(buffer, sizeof(buffer), python_stdout_)) {
        string line(buffer);
        line.erase(line.find_last_not_of(" \n\r\t") + 1);
        
        if (line == "END") {
            break;
        }
        
        if (line == "ERROR") {
            std::cerr << "ERROR: ML server returned error" << std::endl;
            break;
        }
        
        try {
            double score = stod(line);
            importance.push_back(score);
        } catch (...) {
            // ignore parse errors
        }
    }
    
    // check if we got the right number of scores
    if (importance.size() != num_vars) {
        std::cerr << "WARNING: ML server returned " << importance.size() << " scores, expected " << num_vars << ". Using fallback." << std::endl;
        // fallback to heuristics if not
        const auto& occurrences = cnf.get_variable_occurrences();
        importance.clear();
        importance.resize(num_vars, 0.5);
        
        if (!occurrences.empty()) {
            uint32_t max_occ = 0;
            for (uint32_t var = 1; var <= num_vars; var++) {
                uint32_t occ = var < occurrences.size() ? occurrences[var] : 0;
                max_occ = max(max_occ, occ);
            }
            
            if (max_occ > 0) {
                for (uint32_t var = 1; var <= num_vars; var++) {
                    uint32_t occ = var < occurrences.size() ? occurrences[var] : 0;
                    importance[var - 1] = 0.2 + 0.6 * (static_cast<double>(occ) / max_occ);
                }
            }
        }
    }
    
    return importance;
}

// call ML model to get predictions based on features
vector<double> MLHashInterface::call_python_predictor(const CNFFeatures& features) {
    // write features to temporary file
    string temp_file = "/tmp/cnf_features_" + to_string(getpid()) + ".txt";
    ofstream out(temp_file);
    
    if (!out.is_open()) {
        std::cerr << "ERROR: Failed to create temporary feature file" << std::endl;
        return {};
    }
    
    // write features in simple format
    out << features.num_variables << " "
        << features.num_clauses << " "
        << features.avg_clause_size << " "
        << features.variable_clause_ratio << "\n";
    
    // write all variable occurrences for accurate statistics
    for (size_t i = 0; i < features.variable_occurrences.size(); i++) {
        out << features.variable_occurrences[i];
        if (i < features.variable_occurrences.size() - 1) {
            out << " ";
        }
    }
    out << "\n";
    
    out.close();
    
    // call Python script (with stderr to stdout to capture errors)
    string cmd = "python3 ml_model/hash_predictor.py " + temp_file + " 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
    
    if (!pipe) {
        std::cerr << "ERROR: Failed to execute Python predictor" << std::endl;
        remove(temp_file.c_str());
        return {};
    }
    
    // read predictions
    vector<double> predictions;
    char buffer[128];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        try {
            double val = stod(buffer);
            predictions.push_back(val);
        } catch (...) {
            // ignore parse errors
        }
    }
    
    pclose(pipe);
    remove(temp_file.c_str());
    
    return predictions;
}

// generate heuristic-based XOR constraints using variable occurrences (more likely to include high-occurrence variables)
vector<XorConstraint> MLHashInterface::generate_heuristic_hashes(const CNF& cnf, uint32_t num_hashes) {
    const auto& occurrences = cnf.get_variable_occurrences();
    uint32_t num_vars = cnf.num_variables();
    
    // sort variables by occurrence
    vector<pair<uint32_t, uint32_t>> var_freq;
    for (uint32_t var = 1; var <= num_vars; var++) {
        uint32_t occ = var < occurrences.size() ? occurrences[var] : 0;
        var_freq.push_back({occ, var});
    }
    sort(var_freq.begin(), var_freq.end(), greater<pair<uint32_t, uint32_t>>());
    
    // select top variables by frequency
    vector<XorConstraint> constraints;
    constraints.reserve(num_hashes);
    
    double target_sparsity = 0.3;  // target 30% of variables per constraint
    uint32_t target_vars = max(1u, static_cast<uint32_t>(num_vars * target_sparsity));
    
    for (uint32_t i = 0; i < num_hashes; i++) {
        XorConstraint constraint;
        
        for (uint32_t j = 0; j < target_vars && j < num_vars; j++) {
            uint32_t idx = (i * target_vars + j) % num_vars;
            constraint.variables.push_back(var_freq[idx].second);
        }
        
        // RHS based on hash index
        constraint.rhs = (i % 2) == 0;
        constraints.push_back(constraint);
    }
    
    return constraints;
}

} // namespace sharpsat
