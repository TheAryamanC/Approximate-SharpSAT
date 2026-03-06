#ifndef APPROXIMATE_COUNTER_H
#define APPROXIMATE_COUNTER_H

#include "cnf/cnf_structure.h"
#include "xor/xor_hash_generator.h"
#include "xor/ml_hash_interface.h"
#include "solver/sat_solver.h"
#include <memory>
#include <vector>
#include <cmath>

namespace sharpsat {

// Configuration for approximate counting
struct CounterConfig {
    double epsilon;           // Approximation factor (tolerance) - e.g., 0.8 means count is within factor of exp(0.8) ~ 2.2
    double delta;             // Confidence parameter - e.g., 0.2 means 1-0.2 = 0.8 or 80% confidence
    uint32_t seed;            // Random seed
    bool use_ml_hashes;       // Use ML-enhanced hash generation
    bool use_cuda;            // Enable CUDA acceleration
    double timeout_seconds;   // Timeout for each SAT solver call
    uint32_t num_trials;      // Number of trials to run
    
    CounterConfig()
        : epsilon(0.8), delta(0.2), seed(42), use_ml_hashes(false),
          use_cuda(true), timeout_seconds(60.0), num_trials(10) {}
    
    // Calculate cell threshold from epsilon using formula: ceil(4.03 * (1 + 1/epsilon)^2)
    double get_cell_threshold() const {
        double ratio = 1.0 + 1.0 / epsilon;
        return std::ceil(4.03 * ratio * ratio);
    }
};

// Result of approximate counting
struct CountResult {
    double count;             // Estimated model count
    double lower_bound;       // Lower bound on count
    double upper_bound;       // Upper bound on count
    uint32_t num_iterations;  // Number of iterations performed
    double time_seconds;      // Total time taken - compare timing using ML + CUDA
    bool successful;          // Whether counting was successful
    
    CountResult()
        : count(0.0), lower_bound(0.0), upper_bound(0.0),
          num_iterations(0), time_seconds(0.0), successful(false) {}
};

// Approximate model counter
class ApproximateCounter {
public:
    explicit ApproximateCounter(const CounterConfig& config = CounterConfig());
    
    // Main counting interface
    CountResult count(const CNF& cnf);
    
    // Set configuration
    void set_config(const CounterConfig& config) { config_ = config; }
    const CounterConfig& get_config() const { return config_; }
    
private:
    // Core counting algorithm - inspired by ApproxMC
    CountResult approxmc(const CNF& cnf);
    
    // Find hash level (number of XOR constraints) where formula is SAT with reasonable probability
    // This partitions the solution space into ~2^k cells
    uint32_t find_hash_level(const CNF& cnf);
    
    // Check satisfiability with XOR constraints
    bool check_sat_with_xors(const CNF& cnf, const std::vector<XorConstraint>& xors, std::unordered_map<Variable, bool>& assignment);
    
    // Apply XOR constraints via Gaussian elimination
    std::unordered_map<Variable, bool> apply_xor_constraints(const std::vector<XorConstraint>& xors, uint32_t num_variables);
    
    // Compute bounds from hash level
    void compute_bounds(uint32_t hash_level, uint32_t num_variables, CountResult& result);
    
    CounterConfig config_;
    std::unique_ptr<XorHashGenerator> hash_generator_;
    std::unique_ptr<MLHashInterface> ml_interface_;
    std::unique_ptr<SATSolver> sat_solver_;
};

} // namespace sharpsat

#endif // APPROXIMATE_COUNTER_H
