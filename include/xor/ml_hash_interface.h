#ifndef ML_HASH_INTERFACE_H
#define ML_HASH_INTERFACE_H

#include "cnf/cnf_structure.h"
#include "xor_hash_generator.h"
#include <string>
#include <vector>
#include <memory>
#include <random>

namespace sharpsat {

// Features extracted from CNF for ML model
struct CNFFeatures {
    uint32_t num_variables;
    uint32_t num_clauses;
    double avg_clause_size;
    double variable_clause_ratio;
    std::vector<uint32_t> variable_occurrences;
    std::vector<double> variable_balance;  // ratio of positive/negative occurrences
    double max_occurrence;
    double min_occurrence;
    
    CNFFeatures() : num_variables(0), num_clauses(0), avg_clause_size(0.0),
                    variable_clause_ratio(0.0), max_occurrence(0.0), min_occurrence(0.0) {}
};

// ML-based hash generator interface
class MLHashInterface {
public:
    MLHashInterface();
    explicit MLHashInterface(const std::string& model_path);
    ~MLHashInterface();
    
    // Initialize ML model
    bool initialize(const std::string& model_path);
    
    // Check if ML model is available
    bool is_available() const { return model_available_; }
    
    // Extract features from CNF
    static CNFFeatures extract_features(const CNF& cnf);
    
    // Predict optimal hash configuration
    HashConfig predict_hash_config(const CNF& cnf, uint32_t num_hashes);
    
    // Generate ML-enhanced XOR constraints
    std::vector<XorConstraint> generate_ml_hashes(const CNF& cnf, uint32_t num_hashes);
    
    // Predict variable importance scores for XOR generation
    std::vector<double> predict_variable_importance(const CNF& cnf);
    
private:
    bool model_available_;
    std::string model_path_;
    std::unique_ptr<XorHashGenerator> fallback_generator_;
    
    // RNG state for generating hashes
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_dist_;
    std::uniform_int_distribution<int> bool_dist_;
    
    // Persistent Python ML server
    FILE* python_stdin_;
    FILE* python_stdout_;
    FILE* python_stderr_;
    pid_t python_pid_;
    
    // Start/stop persistent Python ML server
    bool start_ml_server();
    void stop_ml_server();
    
    // Call Python ML model via persistent server
    std::vector<double> call_python_predictor(const CNFFeatures& features);
    
    // Generate heuristic-based hashes using variable statistics
    std::vector<XorConstraint> generate_heuristic_hashes(const CNF& cnf, uint32_t num_hashes);
};

} // namespace sharpsat

#endif // ML_HASH_INTERFACE_H
