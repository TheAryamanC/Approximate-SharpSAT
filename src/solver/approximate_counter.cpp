#include "solver/approximate_counter.h"
#include "utils/timer.h"
#include "cuda_interface.h"
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace std;
namespace sharpsat {

// Constructor
ApproximateCounter::ApproximateCounter(const CounterConfig& config) : config_(config) {
    hash_generator_ = make_unique<XorHashGenerator>(config.seed);
    if (config.use_ml_hashes) {
        ml_interface_ = make_unique<MLHashInterface>("src/ml_model/model.pkl");
        ml_interface_->set_seed(config.seed);
    }
    bool use_gpu = sharpsat::cuda::is_cuda_available();
    sat_solver_ = make_unique<SATSolver>(config.timeout_seconds, use_gpu);
}

// Main counting interface
CountResult ApproximateCounter::count(const CNF& cnf) {
    // start timer
    ScopedTimer timer("total_counting");
    
    // run core counting algorithm
    CountResult result = approxmc(cnf);
    
    // stop timer
    result.time_seconds = TimerRegistry::instance().get_elapsed("total_counting");
    
    return result;
}

CountResult ApproximateCounter::approxmc(const CNF& cnf) {
    CountResult result;
    
    // trivial cases - empty formula + formula with empty clause
    if (cnf.is_empty()) {
        result.count = pow(2.0, cnf.num_variables());
        result.lower_bound = result.count;
        result.upper_bound = result.count;
        result.successful = true;
        return result;
    }
    if (cnf.has_empty_clause()) {
        result.count = 0.0;
        result.lower_bound = 0.0;
        result.upper_bound = 0.0;
        result.successful = true;
        return result;
    }
    
    // Initialize CUDA context BEFORE any parallel execution to avoid race conditions
    if (config_.use_cuda) {
        if (!sharpsat::cuda::initialize_cuda_context(0)) {
            cerr << "Warning: Failed to initialize CUDA, falling back to CPU" << endl;
            config_.use_cuda = false;
        }
    }
    
    // find hash level (number of XOR constraints to apply)
    uint32_t hash_level = find_hash_level(cnf);
    
    // if hash level is 0, formula is UNSAT with high probability
    if (hash_level == 0) {
        result.count = 0.0;
        result.lower_bound = 0.0;
        result.upper_bound = 0.0;
        result.successful = true;
        return result;
    }
    
    // run multiple iterations and get SAT count
    uint32_t num_iterations = config_.num_trials;
    uint32_t sat_count = 0;
    
    // run all iterations
    for (uint32_t iter = 0; iter < num_iterations; iter++) {
        // generate XOR constraints depending on configuration
        uint32_t iter_seed = config_.seed + iter;
        XorHashGenerator iter_hash_gen(iter_seed);
        
        vector<XorConstraint> xors;
        if (config_.use_ml_hashes) {
            MLHashInterface iter_ml("src/ml_model/model.pkl");
            iter_ml.set_seed(iter_seed);
            xors = iter_ml.generate_ml_hashes(cnf, hash_level);
        } else {
            double sparsity = XorHashGenerator::get_recommended_sparsity(cnf.num_variables());
            xors = iter_hash_gen.generate_random_hashes(cnf.num_variables(), hash_level, sparsity);
        }
        
        // check satisfiability with XOR constraints
        unordered_map<Variable, bool> assignment;
        if (check_sat_with_xors(cnf, xors, assignment)) {
            sat_count++;
        }
    }
    
    result.num_iterations = num_iterations;
    
    // estimate count based on SAT probability
    double sat_prob = static_cast<double>(sat_count) / num_iterations;
    
    if (sat_prob > 0.0) {
        double cell_threshold = config_.get_cell_threshold();
        double cell_size = pow(2.0, cnf.num_variables() - hash_level);
        
        // use median estimator - adjust count based on observed SAT probability
        result.count = cell_size * (static_cast<double>(num_iterations) / sat_count) * cell_threshold;
        
        // compute bounds
        compute_bounds(hash_level, cnf.num_variables(), result);
        result.successful = true;
    } else {
        // all iterations were UNSAT - count is very small
        result.count = pow(2.0, cnf.num_variables() - hash_level - 1);
        result.lower_bound = 0.0;
        result.upper_bound = result.count * 2.0;
        result.successful = true;
    }
    
    return result;
}

// Find hash level (number of XOR constraints) where formula is SAT with reasonable probability
uint32_t ApproximateCounter::find_hash_level(const CNF& cnf) {
    uint32_t left = 0;
    uint32_t right = cnf.num_variables();
    uint32_t hash_level = 0;
    
    double sparsity = XorHashGenerator::get_recommended_sparsity(cnf.num_variables());
    
    // binary search
    while (left <= right && right <= cnf.num_variables()) {
        uint32_t mid = (left + right) / 2;
        
        // run a few samples to estimate SAT probability at this hash level
        uint32_t num_samples = 5; // small number of samples for quick estimation
        uint32_t sat_samples = 0;
        
        for (uint32_t i = 0; i < num_samples; i++) {
            // generate XOR constraints
            vector<XorConstraint> xors;
            if (config_.use_ml_hashes) {
                xors = ml_interface_->generate_ml_hashes(cnf, mid);
            } else {
                xors = hash_generator_->generate_random_hashes(cnf.num_variables(), mid, sparsity);
            }
            
            // check SAT
            unordered_map<Variable, bool> assignment;
            if (check_sat_with_xors(cnf, xors, assignment)) {
                sat_samples++;
            }
        }
        
        // estimate SAT probability
        double sat_prob = static_cast<double>(sat_samples) / num_samples;
        
        // sat_prob in a reasonable range (not too high, not too low)
        if (sat_prob > 0.8) {
            // too many SAT, increase hash level (add more XOR constraints)
            hash_level = mid;
            left = mid + 1;
        } else if (sat_prob < 0.3) {
            // too few SAT, decrease hash level (remove XOR constraints)
            if (mid == 0) break;
            right = mid - 1;
        } else {
            hash_level = mid;
            break;
        }
    }
    
    return hash_level;
}

// Check if formula is satisfiable with added XOR constraints
bool ApproximateCounter::check_sat_with_xors(const CNF& cnf, const vector<XorConstraint>& xors, unordered_map<Variable, bool>& assignment) {
    // apply XOR constraints via Gaussian elimination
    assignment = apply_xor_constraints(xors, cnf.num_variables());
    
    // apply assignment to CNF and check SAT
    CNF simplified_cnf = cnf.clone();
    simplified_cnf.apply_assignment(assignment);
    
    if (simplified_cnf.has_empty_clause()) {
        return false;
    }
    
    // use SAT solver on simplified formula
    return sat_solver_->solve(simplified_cnf, assignment);
}

// Apply XOR constraints via Gaussian elimination
unordered_map<Variable, bool> ApproximateCounter::apply_xor_constraints(const vector<XorConstraint>& xors, uint32_t num_variables) {
    unordered_map<Variable, bool> assignment;
    
    if (xors.empty()) { // no constraints, return empty assignment
        return assignment;
    }
    
    // if CUDA is enabled, use GPU for Gaussian elimination, otherwise use CPU
    if (config_.use_cuda) {
        vector<uint32_t> flat_vars, offsets;
        vector<uint8_t> rhs;
        sharpsat::cuda::convert_xors_to_gpu_format(xors, flat_vars, offsets, rhs);
        
        // GPU version - defined elsewhere
        bool success = sharpsat::cuda::gaussian_elimination_gpu(flat_vars, offsets, rhs, num_variables, assignment);
    } else {
        // CPU version - simple Gaussian elimination
        vector<vector<bool>> matrix;
        vector<bool> rhs_vec;
        
        // build augmented matrix
        for (const auto& xor_c : xors) {
            vector<bool> row(num_variables, false);
            for (Variable v : xor_c.variables) {
                if (v > 0 && v <= num_variables) {
                    row[v - 1] = true;
                }
            }
            matrix.push_back(row);
            rhs_vec.push_back(xor_c.rhs);
        }
        
        // row reduce to echelon form
        size_t num_rows = matrix.size();
        size_t pivot_row = 0;
        
        for (size_t col = 0; col < num_variables && pivot_row < num_rows; col++) {
            // find pivot
            size_t pivot = pivot_row;
            for (size_t r = pivot_row; r < num_rows; r++) {
                if (matrix[r][col]) {
                    pivot = r;
                    break;
                }
            }
            
            if (!matrix[pivot][col]) {
                continue;  // no pivot in this column, move to next column
            }
            
            // swap pivot row to current row
            if (pivot != pivot_row) {
                swap(matrix[pivot], matrix[pivot_row]);
                swap(rhs_vec[pivot], rhs_vec[pivot_row]);
            }
            
            // eliminate all rows below pivot
            for (size_t r = pivot_row + 1; r < num_rows; r++) {
                if (matrix[r][col]) {
                    for (size_t c = 0; c < num_variables; c++) {
                        matrix[r][c] = matrix[r][c] != matrix[pivot_row][c];
                    }
                    rhs_vec[r] = rhs_vec[r] != rhs_vec[pivot_row];
                }
            }
            
            pivot_row++;
        }
        
        // get assignment from reduced matrix
        for (int r = static_cast<int>(num_rows) - 1; r >= 0; r--) {
            // find leading variable in this row
            int lead_var = -1;
            for (size_t c = 0; c < num_variables; c++) {
                if (matrix[r][c]) {
                    lead_var = c;
                    break;
                }
            }
            
            if (lead_var >= 0) {
                bool value = rhs_vec[r];
                // XOR with already assigned variables
                for (size_t c = lead_var + 1; c < num_variables; c++) {
                    if (matrix[r][c]) {
                        Variable v = c + 1;
                        if (assignment.find(v) != assignment.end()) {
                            value = value != assignment[v];
                        }
                    }
                }
                assignment[lead_var + 1] = value;
            }
        }
    }
    
    return assignment;
}

void ApproximateCounter::compute_bounds(uint32_t hash_level, uint32_t num_variables, CountResult& result) {
    // compute confidence bounds based on epsilon and delta
    double factor = 1.0 + config_.epsilon;    
    result.lower_bound = result.count / factor;
    result.upper_bound = result.count * factor;
}

} // namespace sharpsat
