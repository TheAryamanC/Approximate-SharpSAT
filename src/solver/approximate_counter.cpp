#include "solver/approximate_counter.h"
#include "utils/timer.h"
#include "cuda_interface.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <thread>
#include <future>
#include <atomic>

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
    static constexpr double LOG10_2 = 0.30102999566398119521;
    static constexpr double MAX_SAFE_LOG10 = 307.0;
    
    if (cnf.is_empty()) {
        // All 2^n assignments satisfy an empty formula.
        // Use exact pow(2,n) for n<=1023 (fits in double), log-space otherwise.
        result.log10_count = cnf.num_variables() * LOG10_2;
        result.count = (cnf.num_variables() <= 1023)
                     ? std::pow(2.0, static_cast<double>(cnf.num_variables()))
                     : std::numeric_limits<double>::infinity();
        result.lower_bound = result.count;
        result.upper_bound = result.count;
        result.log10_lower_bound = result.log10_count;
        result.log10_upper_bound = result.log10_count;
        result.successful = true;
        return result;
    }
    if (cnf.has_empty_clause()) {
        // No satisfying assignment exists.
        result.count = 0.0;
        result.lower_bound = 0.0;
        result.upper_bound = 0.0;
        result.log10_count       = -std::numeric_limits<double>::infinity();
        result.log10_lower_bound = -std::numeric_limits<double>::infinity();
        result.log10_upper_bound = -std::numeric_limits<double>::infinity();
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
        result.log10_count       = -std::numeric_limits<double>::infinity();
        result.log10_lower_bound = -std::numeric_limits<double>::infinity();
        result.log10_upper_bound = -std::numeric_limits<double>::infinity();
        result.successful = true;
        return result;
    }
    
    // run multiple iterations and get SAT count
    uint32_t num_iterations = config_.num_trials;
    
    // Pre-generate all XOR constraint sets (sequential; ensures reproducibility
    // and allows ML interface to remain single-threaded)
    double sparsity = XorHashGenerator::get_recommended_sparsity(cnf.num_variables());
    std::vector<std::vector<XorConstraint>> all_xors(num_iterations);
    for (uint32_t iter = 0; iter < num_iterations; iter++) {
        uint32_t iter_seed = config_.seed + iter;
        if (config_.use_ml_hashes) {
            ml_interface_->set_seed(iter_seed);
            all_xors[iter] = ml_interface_->generate_ml_hashes(cnf, hash_level);
        } else {
            XorHashGenerator iter_hash_gen(iter_seed);
            all_xors[iter] = iter_hash_gen.generate_random_hashes(
                cnf.num_variables(), hash_level, sparsity);
        }
    }
    
    // Determine number of worker threads for fork-join parallelism
    uint32_t hw_threads = std::thread::hardware_concurrency();
    uint32_t num_threads = (config_.num_threads > 0) ? config_.num_threads
                                                      : std::max(1u, hw_threads);
    num_threads = std::min(num_threads, num_iterations);
    
    // Fork: launch one async task per partition
    std::vector<std::future<uint32_t>> futures;
    futures.reserve(num_threads);
    
    for (uint32_t tid = 0; tid < num_threads; tid++) {
        uint32_t start = (tid * num_iterations) / num_threads;
        uint32_t end   = ((tid + 1) * num_iterations) / num_threads;
        if (start >= end) continue;
        
        futures.push_back(std::async(std::launch::async,
            [this, &cnf, &all_xors, start, end]() -> uint32_t {
                uint32_t local_sat = 0;
                // Each worker owns its own SATSolver — no shared mutable state
                SATSolver local_solver(config_.timeout_seconds, false);
                for (uint32_t iter = start; iter < end; iter++) {
                    auto assignment = apply_xor_constraints_cpu(
                        all_xors[iter], cnf.num_variables());
                    CNF simplified = cnf.clone();
                    simplified.apply_assignment(assignment);
                    if (!simplified.has_empty_clause() &&
                        local_solver.solve(simplified, assignment)) {
                        local_sat++;
                    }
                }
                return local_sat;
            }));
    }
    
    // Join: collect results from all workers
    uint32_t sat_count = 0;
    for (auto& f : futures) {
        sat_count += f.get();
    }
    
    result.num_iterations = num_iterations;
    
    // Compute count in log-space to handle extremely large formulas without overflow.
    // All arithmetic is done as log10 values; result.count is set from them
    // (it will be std::numeric_limits<double>::infinity() when the formula is huge).
    const double n = static_cast<double>(cnf.num_variables());
    const double k = static_cast<double>(hash_level);
    
    if (sat_count > 0) {
        double cell_threshold = config_.get_cell_threshold();
        double ni = static_cast<double>(num_iterations);
        double sc = static_cast<double>(sat_count);
        
        // log10(count) = (n-k)*log10(2) + log10(ni/sc) + log10(cell_threshold)
        result.log10_count = (n - k) * LOG10_2
                           + std::log10(ni) - std::log10(sc)
                           + std::log10(cell_threshold);
        
        result.count = (result.log10_count <= MAX_SAFE_LOG10)
                     ? std::pow(10.0, result.log10_count)
                     : std::numeric_limits<double>::infinity();
        
        // compute bounds
        compute_bounds(hash_level, cnf.num_variables(), result);
        result.successful = true;
    } else {
        // all iterations were UNSAT — count is very small
        result.log10_count = (n - k - 1.0) * LOG10_2;
        result.count = (result.log10_count <= MAX_SAFE_LOG10)
                     ? std::pow(10.0, result.log10_count)
                     : std::numeric_limits<double>::infinity();
        result.log10_lower_bound = -std::numeric_limits<double>::infinity();
        result.lower_bound = 0.0;
        result.log10_upper_bound = result.log10_count + LOG10_2; // *2
        result.upper_bound = (result.log10_upper_bound <= MAX_SAFE_LOG10)
                           ? std::pow(10.0, result.log10_upper_bound)
                           : std::numeric_limits<double>::infinity();
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
                ml_interface_->set_seed(config_.seed + mid * 1000 + i);
                xors = ml_interface_->generate_ml_hashes(cnf, mid);
            } else {
                hash_generator_->set_seed(config_.seed + mid * 1000 + i);
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

// CPU-only Gaussian elimination over GF(2) — thread-safe static implementation
// used by parallel trial workers so they never touch shared GPU state.
unordered_map<Variable, bool> ApproximateCounter::apply_xor_constraints_cpu(
        const vector<XorConstraint>& xors, uint32_t num_variables) {
    unordered_map<Variable, bool> assignment;
    
    if (xors.empty()) {
        return assignment;
    }
    
    // Build augmented matrix [A | b] where each row encodes one XOR constraint
    vector<vector<bool>> matrix;
    vector<bool> rhs_vec;
    
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
    
    // Forward elimination to row echelon form
    size_t num_rows = matrix.size();
    size_t pivot_row = 0;
    
    for (size_t col = 0; col < num_variables && pivot_row < num_rows; col++) {
        // Find pivot in this column
        size_t pivot = pivot_row;
        for (size_t r = pivot_row; r < num_rows; r++) {
            if (matrix[r][col]) { pivot = r; break; }
        }
        
        if (!matrix[pivot][col]) continue; // no pivot in this column
        
        if (pivot != pivot_row) {
            swap(matrix[pivot], matrix[pivot_row]);
            swap(rhs_vec[pivot], rhs_vec[pivot_row]);
        }
        
        // Eliminate rows below pivot
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
    
    // Back-substitution
    for (int r = static_cast<int>(num_rows) - 1; r >= 0; r--) {
        int lead_var = -1;
        for (size_t c = 0; c < num_variables; c++) {
            if (matrix[r][c]) { lead_var = static_cast<int>(c); break; }
        }
        if (lead_var < 0) continue;
        
        bool value = rhs_vec[r];
        for (size_t c = static_cast<size_t>(lead_var) + 1; c < num_variables; c++) {
            if (matrix[r][c]) {
                Variable v = static_cast<Variable>(c + 1);
                auto it = assignment.find(v);
                if (it != assignment.end()) {
                    value = value != it->second;
                }
            }
        }
        assignment[static_cast<Variable>(lead_var + 1)] = value;
    }
    
    return assignment;
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
    // Compute confidence bounds in log-space (avoids overflow for huge formulas)
    static constexpr double MAX_SAFE_LOG10 = 307.0;
    double log10_factor = std::log10(1.0 + config_.epsilon);
    
    result.log10_lower_bound = result.log10_count - log10_factor;
    result.log10_upper_bound = result.log10_count + log10_factor;
    
    result.lower_bound = (result.log10_lower_bound <= MAX_SAFE_LOG10)
                       ? std::pow(10.0, result.log10_lower_bound)
                       : std::numeric_limits<double>::infinity();
    result.upper_bound = (result.log10_upper_bound <= MAX_SAFE_LOG10)
                       ? std::pow(10.0, result.log10_upper_bound)
                       : std::numeric_limits<double>::infinity();
}

} // namespace sharpsat
