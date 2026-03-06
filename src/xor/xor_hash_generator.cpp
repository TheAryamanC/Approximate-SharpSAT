#include "xor/xor_hash_generator.h"
#include <cmath>

using namespace std;
namespace sharpsat {

// constructors
XorHashGenerator::XorHashGenerator(uint32_t seed) 
    : rng_(seed), uniform_dist_(0.0, 1.0), bool_dist_(0, 1) {
}

void XorHashGenerator::set_seed(uint32_t seed) {
    rng_.seed(seed);
}

// generate random sparse XOR constraints
vector<XorConstraint> XorHashGenerator::generate_random_hashes(uint32_t num_variables, uint32_t num_hashes, double sparsity) {
    vector<XorConstraint> constraints;
    constraints.reserve(num_hashes);
    
    for (uint32_t i = 0; i < num_hashes; i++) {
        constraints.push_back(generate_single_hash(num_variables, sparsity));
    }
    
    return constraints;
}

// generate a single random XOR constraint
XorConstraint XorHashGenerator::generate_single_hash(
    uint32_t num_variables,
    double sparsity) {
    
    XorConstraint constraint;
    
    // randomly select variables based on sparsity
    for (uint32_t var = 1; var <= num_variables; var++) {
        if (uniform_dist_(rng_) < sparsity) {
            constraint.variables.push_back(var);
        }
    }
    
    // ensure at least one variable
    if (constraint.variables.empty() && num_variables > 0) {
        uniform_int_distribution<uint32_t> var_dist(1, num_variables);
        constraint.variables.push_back(var_dist(rng_));
    }
    
    // random RHS (0 or 1)
    constraint.rhs = bool_dist_(rng_) == 1;
    
    return constraint;
}

// get recommended sparsity for given problem size (heuristic)
double XorHashGenerator::get_recommended_sparsity(uint32_t num_variables) {
    if (num_variables <= 60) return 0.5; 
    // slowly decay sparsity as variables increase
    return std::max(0.2, 0.5 * (log2(60) / log2(num_variables)));
}

} // namespace sharpsat
