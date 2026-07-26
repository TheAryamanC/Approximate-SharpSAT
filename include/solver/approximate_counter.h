#ifndef APPROXIMATE_COUNTER_H
#define APPROXIMATE_COUNTER_H

#include "cnf/cnf_structure.h"
#include "xor/xor_hash_generator.h"
#include "xor/ml_hash_interface.h"
#include "solver/sat_solver.h"
#include <memory>
#include <vector>
#include <cmath>
#include <future>
#include <thread>
#include <limits>

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
    uint32_t num_threads;     // CPU threads for parallel trials (0 = auto-detect)
    
    CounterConfig()
        : epsilon(0.8), delta(0.2), seed(42), use_ml_hashes(false),
          use_cuda(true), timeout_seconds(60.0), num_trials(10), num_threads(0) {}
    
    // Calculate cell threshold from epsilon using formula: ceil(4.03 * (1 + 1/epsilon)^2)
    double get_cell_threshold() const {
        double ratio = 1.0 + 1.0 / epsilon;
        return std::ceil(4.03 * ratio * ratio);
    }
};

// Result of approximate counting
struct CountResult {
    double count;             // Estimated model count (may be inf for huge formulas)
    double lower_bound;       // Lower bound on count (may be inf for huge formulas)
    double upper_bound;       // Upper bound on count (may be inf for huge formulas)
    double log10_count;       // log10 of model count — accurate even when count overflows
    double log10_lower_bound; // log10 of lower bound
    double log10_upper_bound; // log10 of upper bound
    uint32_t num_iterations;  // Number of iterations performed
    double time_seconds;      // Total time taken - compare timing using ML + CUDA
    bool successful;          // Whether counting was successful
    
    // Returns the base-10 order of magnitude: floor(log10(count))
    // e.g. a formula with ~10^301 solutions returns 301
    int64_t order_of_magnitude() const {
        if (!std::isfinite(log10_count) || log10_count < 0.0) return 0;
        return static_cast<int64_t>(std::floor(log10_count));
    }
    
    CountResult()
        : count(0.0), lower_bound(0.0), upper_bound(0.0),
          log10_count(-std::numeric_limits<double>::infinity()),
          log10_lower_bound(-std::numeric_limits<double>::infinity()),
          log10_upper_bound(-std::numeric_limits<double>::infinity()),
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
    
    // Apply XOR constraints via Gaussian elimination (may use CUDA; not thread-safe)
    std::unordered_map<Variable, bool> apply_xor_constraints(const std::vector<XorConstraint>& xors, uint32_t num_variables);
    
    // CPU-only Gaussian elimination — thread-safe, used by parallel trial workers
    static std::unordered_map<Variable, bool> apply_xor_constraints_cpu(
        const std::vector<XorConstraint>& xors, uint32_t num_variables);
    
    // Compute bounds from hash level
    void compute_bounds(uint32_t hash_level, uint32_t num_variables, CountResult& result);
    
    CounterConfig config_;
    std::unique_ptr<XorHashGenerator> hash_generator_;
    std::unique_ptr<MLHashInterface> ml_interface_;
    std::unique_ptr<SATSolver> sat_solver_;
};

} // namespace sharpsat

#endif // APPROXIMATE_COUNTER_H
