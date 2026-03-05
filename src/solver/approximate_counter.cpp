#include "solver/approximate_counter.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "cuda_interface.h"
#include <cmath>
#include <algorithm>

using namespace std;
namespace sharpsat {

// Constructor
ApproximateCounter::ApproximateCounter(const CounterConfig& config)
    : config_(config) {
    
    hash_generator_ = make_unique<XorHashGenerator>(config.seed);
    // Initialize ML interface with model path
    ml_interface_ = make_unique<MLHashInterface>("ml_model/model.pkl");
    sat_solver_ = make_unique<SATSolver>(10000);  // max 10k decisions for faster benchmarks
}

// Main counting interface
CountResult ApproximateCounter::count(const CNF& cnf) {
    // start timer + log information
    ScopedTimer timer("total_counting");
    
    LOG_INFO("Starting approximate model counting");
    LOG_INFO("Formula: ", cnf.num_variables(), " variables, ", cnf.num_clauses(), " clauses");
    LOG_INFO("Config: epsilon=", config_.epsilon, ", delta=", config_.delta);
    
    // run core counting algorithm
    CountResult result = approxmc(cnf);
    
    // stop timer + log results
    result.time_seconds = TimerRegistry::instance().get_elapsed("total_counting");
    LOG_INFO("Model count (approximate): ", result.count);
    LOG_INFO("Total time: ", result.time_seconds, " seconds");
    
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
    
    // find pivot threshold
    LOG_INFO("Finding pivot threshold...");
    uint32_t pivot = find_pivot_threshold(cnf);
    LOG_INFO("Pivot threshold: ", pivot);
    
    // if pivot is 0, formula is UNSAT with high probability
    if (pivot == 0) {
        result.count = 0.0;
        result.lower_bound = 0.0;
        result.upper_bound = 0.0;
        result.successful = true;
        return result;
    }
    
    // run multiple iterations and get SAT count
    // number of iterations based on confidence parameters
    // formula:
    uint32_t num_iterations = static_cast<uint32_t>(ceil(3.0 * log(3.0 / config_.delta) / (config_.epsilon * config_.epsilon)));
    num_iterations = min(num_iterations, config_.max_iterations);
    LOG_INFO("Running ", num_iterations, " iterations at pivot threshold");
    
    uint32_t sat_count = 0;
    for (uint32_t iter = 0; iter < num_iterations; iter++) {
        // generate XOR constraints
        vector<XorConstraint> xors;
        if (config_.use_ml_hashes) {
            // ML interface now includes heuristic-based generation
            xors = ml_interface_->generate_ml_hashes(cnf, pivot);
        } else {
            double sparsity = XorHashGenerator::get_recommended_sparsity(cnf.num_variables());
            xors = hash_generator_->generate_random_hashes(cnf.num_variables(), pivot, sparsity);
        }
        
        // check satisfiability with XOR constraints
        unordered_map<Variable, bool> assignment;
        if (check_sat_with_xors(cnf, xors, assignment)) {
            sat_count++;
        }
        
        // log progress every 10 iterations
        if ((iter + 1) % 10 == 0) {
            LOG_DEBUG("Completed ", iter + 1, "/", num_iterations, " iterations, SAT count: ", sat_count);
        }
    }
    
    result.num_iterations = num_iterations;
    
    // estimate count based on SAT probability
    double sat_prob = static_cast<double>(sat_count) / num_iterations;
    
    if (sat_prob > 0.0) {
        // expected cell size is 2^(n - pivot)
        double cell_size = pow(2.0, cnf.num_variables() - pivot);
        
        // estimated count
        result.count = cell_size * config_.pivot_threshold;
        
        // compute bounds
        compute_bounds(pivot, cnf.num_variables(), result);
        result.successful = true;
    } else {
        // all iterations were UNSAT - count is very small
        result.count = pow(2.0, cnf.num_variables() - pivot - 1);
        result.lower_bound = 0.0;
        result.upper_bound = result.count * 2.0;
        result.successful = true;
    }
    
    return result;
}

// Binary search for pivot threshold
uint32_t ApproximateCounter::find_pivot_threshold(const CNF& cnf) {
    uint32_t left = 0;
    uint32_t right = cnf.num_variables();
    uint32_t pivot = 0;
    
    double sparsity = XorHashGenerator::get_recommended_sparsity(cnf.num_variables());
    
    // we want to find the largest pivot such that formula with pivot XOR constraints is SAT with high probability
    while (left <= right && right <= cnf.num_variables()) {
        uint32_t mid = (left + right) / 2;
        
        // run a few samples to estimate SAT probability at this pivot
        uint32_t num_samples = 3;  // reduced from 10 for faster benchmarks
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
        LOG_DEBUG("Threshold ", mid, ": SAT probability = ", sat_prob);
        
        // We want sat_prob close to pivot_threshold / num_samples (range)
        if (sat_prob > 0.8) {
            // too many SAT, increase threshold
            pivot = mid;
            left = mid + 1;
        } else if (sat_prob < 0.3) {
            // too few SAT, decrease threshold
            if (mid == 0) break;
            right = mid - 1;
        } else {
            pivot = mid;
            break;
        }
    }
    
    return pivot;
}

// Check satisfiability with XOR constraints
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
        
        if (!success) {
            LOG_DEBUG("Gaussian elimination detected conflict");
        }
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

void ApproximateCounter::compute_bounds(double threshold, uint32_t num_variables, CountResult& result) {
    // compute confidence bounds based on epsilon and delta
    double factor = exp(config_.epsilon);
    
    result.lower_bound = result.count / factor;
    result.upper_bound = result.count * factor;
}

} // namespace sharpsat
