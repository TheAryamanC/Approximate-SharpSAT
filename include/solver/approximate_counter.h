#ifndef APPROXIMATE_COUNTER_H
#define APPROXIMATE_COUNTER_H

#include "cnf/cnf_structure.h"
#include "xor/xor_hash_generator.h"
#include "xor/ml_hash_interface.h"
#include "solver/sat_solver.h"
#include <memory>
#include <vector>

namespace sharpsat {

// Configuration for approximate counting
struct CounterConfig {
    double epsilon;           // Approximation factor (tolerance) - e.g., 0.8 means count is within factor of exp(0.8) ~ 2.2
    double delta;             // Confidence parameter - e.g., 0.2 means 80% confidence
    uint32_t seed;            // Random seed
    bool use_ml_hashes;       // Use ML-enhanced hash generation -- to see if any benefit by running a 2x2 grid of trials
    bool use_cuda;            // Enable CUDA acceleration -- to see if any benefit by running a 2x2 grid of trials
    double pivot_threshold;   // Threshold for satisfiability checks
    uint32_t max_iterations;  // Maximum iterations per threshold
    
    CounterConfig()
        : epsilon(0.8), delta(0.2), seed(42), use_ml_hashes(false),
          use_cuda(true), pivot_threshold(9.0), max_iterations(10) {}
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
    
    // Find pivot threshold m such that adding m XOR constraints - formula is SAT with high probability
    uint32_t find_pivot_threshold(const CNF& cnf);
    
    // Check satisfiability with XOR constraints
    bool check_sat_with_xors(const CNF& cnf, const std::vector<XorConstraint>& xors, std::unordered_map<Variable, bool>& assignment);
    
    // Apply XOR constraints via Gaussian elimination
    std::unordered_map<Variable, bool> apply_xor_constraints(const std::vector<XorConstraint>& xors, uint32_t num_variables);
    
    // Compute bounds from threshold
    void compute_bounds(double threshold, uint32_t num_variables, CountResult& result);
    
    CounterConfig config_;
    std::unique_ptr<XorHashGenerator> hash_generator_;
    std::unique_ptr<MLHashInterface> ml_interface_;
    std::unique_ptr<SATSolver> sat_solver_;
};

} // namespace sharpsat

#endif // APPROXIMATE_COUNTER_H
