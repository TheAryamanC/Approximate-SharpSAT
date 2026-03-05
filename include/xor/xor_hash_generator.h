#ifndef XOR_HASH_GENERATOR_H
#define XOR_HASH_GENERATOR_H

#include "cnf/cnf_structure.h"
#include <vector>
#include <random>
#include <memory>

namespace sharpsat {

// Configuration for XOR hash generation
struct HashConfig {
    double sparsity;           // Probability of a variable appearing in XOR
    uint32_t num_hashes;       // Number of XOR constraints to generate
    uint32_t seed;             // Random seed
    bool use_ml_predictor;     // Use ML model for hash generation
    
    HashConfig() 
        : sparsity(0.5), num_hashes(1), seed(42), use_ml_predictor(false) {}
};

// Generates XOR constraints for universal hashing
class XorHashGenerator {
public:
    explicit XorHashGenerator(uint32_t seed = 42);
    
    // Generate random sparse XOR constraints
    std::vector<XorConstraint> generate_random_hashes(uint32_t num_variables, uint32_t num_hashes, double sparsity);
    
    // Generate a single random XOR constraint
    XorConstraint generate_single_hash(uint32_t num_variables, double sparsity);
    
    // Set random seed
    void set_seed(uint32_t seed);
    
    // Get recommended sparsity for given problem size
    static double get_recommended_sparsity(uint32_t num_variables);
    
private:
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_dist_;
    std::uniform_int_distribution<int> bool_dist_;
};

// Cell structure for approximate counting
struct Cell {
    std::vector<XorConstraint> xor_constraints;
    uint32_t threshold;  // Number of XOR constraints
    
    Cell() : threshold(0) {}
    explicit Cell(uint32_t t) : threshold(t) {}
    
    // Get expected cell size: 2^(n - threshold) where n is num_variables
    double expected_size(uint32_t num_variables) const {
        if (threshold > num_variables) return 0.0;
        return std::pow(2.0, num_variables - threshold);
    }
};

} // namespace sharpsat

#endif // XOR_HASH_GENERATOR_H
