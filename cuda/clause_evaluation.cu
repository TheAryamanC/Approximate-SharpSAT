#ifdef __CUDACC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include "clause_evaluation.cuh"
#include "gpu_utils.cuh"
#include <vector>
#include <algorithm>

using namespace std;
namespace sharpsat {
namespace cuda {

namespace internal {

// kernel to evaluate if a single clause is satisfied under the current assignment
__device__ bool evaluate_clause(const int32_t* literals, uint32_t clause_size, const uint32_t* assignment_vars, const uint8_t* assignment_values, uint32_t num_assignments) {
    // clause is satisfied if at least one literal is true
    for (uint32_t i = 0; i < clause_size; i++) {
        int32_t lit = literals[i];
        uint32_t var = (lit > 0) ? lit : -lit;
        bool sign = (lit > 0);
        
        // check if variable is assigned
        for (uint32_t j = 0; j < num_assignments; j++) {
            if (assignment_vars[j] == var) {
                if ((assignment_values[j] != 0) == sign) {
                    return true;  // literal is satisfied
                }
                break;
            }
        }
    }
    
    return false;  // no literal satisfied (or all false/unassigned)
}

// kernel to check all clauses in parallel
__global__ void check_all_clauses_kernel(const int32_t* clause_lits, const uint32_t* clause_offsets, uint32_t num_clauses, const uint32_t* assignment_vars, const uint8_t* assignment_values, uint32_t num_assignments, uint8_t* clause_satisfied) {
    int clause_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (clause_id >= num_clauses) return; // one thread per clause
    
    uint32_t start = clause_offsets[clause_id];
    uint32_t end = clause_offsets[clause_id + 1];
    uint32_t clause_size = end - start;
    
    const int32_t* clause_literals = &clause_lits[start];
    
    // check if clause has any satisfied literal
    bool satisfied = false;
    bool has_unknown = false;
    
    for (uint32_t i = 0; i < clause_size; i++) {
        int32_t lit = clause_literals[i];
        uint32_t var = (lit > 0) ? lit : -lit;
        bool sign = (lit > 0);
        
        bool var_assigned = false;
        bool var_value = false;
        
        // check if variable is assigned
        for (uint32_t j = 0; j < num_assignments; j++) {
            if (assignment_vars[j] == var) {
                var_assigned = true;
                var_value = (assignment_values[j] != 0);
                break;
            }
        }
        
        // if variable is assigned and satisfies the literal, clause is satisfied
        if (var_assigned) {
            if (var_value == sign) {
                satisfied = true;
                break;
            }
        } else {
            has_unknown = true;
        }
    }
    
    // clause satisfied if it has a true literal
    // clause not satisfied if all literals are assigned false
    // otherwise unknown
    clause_satisfied[clause_id] = (satisfied || has_unknown) ? 1 : 0;
}

// kernel to reduce clause satisfaction results to a single boolean indicating if all clauses are satisfied
__global__ void reduce_satisfaction_kernel(const uint8_t* clause_satisfied, uint32_t num_clauses, uint8_t* all_satisfied) {
    __shared__ uint8_t block_result[256];
    
    int tid = threadIdx.x;
    int clause_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // load result
    uint8_t satisfied = (clause_id < num_clauses) ? clause_satisfied[clause_id] : 1;
    block_result[tid] = satisfied;
    __syncthreads();
    
    // reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            block_result[tid] = block_result[tid] && block_result[tid + s];
        }
        __syncthreads();
    }
    
    // write block result
    if (tid == 0) {
        atomicAnd((int*)all_satisfied, (int)block_result[0]);
    }
}

// kernel to find unit clauses (clauses with exactly one unassigned literal and all others false)
__global__ void find_unit_clauses_kernel(const int32_t* clause_lits, const uint32_t* clause_offsets, uint32_t num_clauses, const uint32_t* assignment_vars, const uint8_t* assignment_values, uint32_t num_assignments, int32_t* unit_literals, uint32_t* num_units) {
    int clause_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (clause_id >= num_clauses) return; // one thread per clause
    
    uint32_t start = clause_offsets[clause_id];
    uint32_t end = clause_offsets[clause_id + 1];
    uint32_t clause_size = end - start;
    
    const int32_t* clause_literals = &clause_lits[start];
    
    // count unassigned literals
    uint32_t unassigned_count = 0;
    int32_t unassigned_lit = 0;
    bool has_satisfied = false;
    
    for (uint32_t i = 0; i < clause_size; i++) {
        int32_t lit = clause_literals[i];
        uint32_t var = (lit > 0) ? lit : -lit;
        bool sign = (lit > 0);
        
        bool var_assigned = false;
        bool var_value = false;
        
        // check if variable is assigned
        for (uint32_t j = 0; j < num_assignments; j++) {
            if (assignment_vars[j] == var) {
                var_assigned = true;
                var_value = (assignment_values[j] != 0);
                break;
            }
        }
        
        // if variable is assigned and satisfies the literal, clause is satisfied
        if (var_assigned) {
            if (var_value == sign) {
                has_satisfied = true;
                break;
            }
        } else {
            unassigned_count++;
            unassigned_lit = lit;
        }
    }
    
    // if exactly one unassigned and not satisfied, it's a unit clause
    if (!has_satisfied && unassigned_count == 1) {
        uint32_t idx = atomicAdd(num_units, 1);
        unit_literals[idx] = unassigned_lit;
    }
}

} // namespace internal

// utility function to convert clauses to flat format for GPU
void convert_clauses_to_gpu_format(const vector<vector<int32_t>>& clauses, vector<int32_t>& flat_lits, vector<uint32_t>& offsets) {
    // pass by reference to avoid unnecessary copying - so need to clear before filling
    flat_lits.clear();
    offsets.clear();
    
    offsets.push_back(0);
    
    // flatten clauses and compute offsets
    for (const auto& clause : clauses) {
        for (int32_t lit : clause) {
            flat_lits.push_back(lit);
        }
        offsets.push_back(flat_lits.size());
    }
}

// utility function to evaluate if all clauses are satisfied under the current assignment
bool evaluate_clauses_gpu(const vector<int32_t>& clause_lits, const vector<uint32_t>& clause_offsets, const unordered_map<uint32_t, bool>& assignment, uint32_t num_variables) {
    if (clause_offsets.size() <= 1) {
        return true;  // no clauses
    }
    
    uint32_t num_clauses = clause_offsets.size() - 1;
    
    // convert assignment to arrays
    vector<uint32_t> vars;
    vector<bool> values;
    
    // convert unordered_map to parallel arrays for GPU
    for (const auto& pair : assignment) {
        vars.push_back(pair.first);
        values.push_back(pair.second);
    }
    
    if (vars.empty()) {
        return true;  // no assignment yet
    }
    
    // allocate GPU memory
    DeviceArray<int32_t> d_clause_lits(clause_lits.size());
    DeviceArray<uint32_t> d_clause_offsets(clause_offsets.size());
    DeviceArray<uint32_t> d_vars(vars.size());
    DeviceArray<uint8_t> d_values(values.size());
    DeviceArray<uint8_t> d_clause_satisfied(num_clauses);
    DeviceArray<uint8_t> d_all_satisfied(1);
    
    // copy to GPU
    d_clause_lits.copy_from_host(clause_lits.data(), clause_lits.size());
    d_clause_offsets.copy_from_host(clause_offsets.data(), clause_offsets.size());
    d_vars.copy_from_host(vars.data(), vars.size());
    
    // convert bool to uint8_t
    vector<uint8_t> values_u8(values.size());
    for (size_t i = 0; i < values.size(); i++) {
        values_u8[i] = values[i] ? 1 : 0;
    }
    d_values.copy_from_host(values_u8.data(), values_u8.size());
    
    uint8_t all_sat = 1;
    d_all_satisfied.copy_from_host(&all_sat, 1);
    
    // launch kernel
    int block_size = 256;
    int num_blocks = (num_clauses + block_size - 1) / block_size;
    
    internal::check_all_clauses_kernel<<<num_blocks, block_size>>>(
        d_clause_lits.data(), d_clause_offsets.data(), num_clauses,
        d_vars.data(), d_values.data(), vars.size(),
        d_clause_satisfied.data()
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // reduce
    internal::reduce_satisfaction_kernel<<<num_blocks, block_size>>>(
        d_clause_satisfied.data(), num_clauses, d_all_satisfied.data()
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // get result
    d_all_satisfied.copy_to_host(&all_sat, 1);
    
    return all_sat != 0;
}

// utility function to check if CNF is satisfied under the current assignment
bool check_cnf_satisfiability_gpu(const vector<int32_t>& clause_lits, const vector<uint32_t>& clause_offsets, const uint32_t* d_assignment_vars, const uint8_t* d_assignment_values, uint32_t num_assignments, uint32_t num_clauses) {
    if (clause_offsets.size() <= 1) {
        return true; // no clauses
    }
    
    // allocate GPU memory
    DeviceArray<int32_t> d_clause_lits(clause_lits.size());
    DeviceArray<uint32_t> d_clause_offsets(clause_offsets.size());
    DeviceArray<uint8_t> d_clause_satisfied(num_clauses);
    DeviceArray<uint8_t> d_all_satisfied(1);
    
    // copy to GPU
    d_clause_lits.copy_from_host(clause_lits.data(), clause_lits.size());
    d_clause_offsets.copy_from_host(clause_offsets.data(), clause_offsets.size());
    
    uint8_t all_sat = 1;
    d_all_satisfied.copy_from_host(&all_sat, 1);
    
    // launch kernels
    int block_size = 256;
    int num_blocks = (num_clauses + block_size - 1) / block_size;
    
    // check clauses
    internal::check_all_clauses_kernel<<<num_blocks, block_size>>>(
        d_clause_lits.data(), d_clause_offsets.data(), num_clauses,
        d_assignment_vars, d_assignment_values, num_assignments,
        d_clause_satisfied.data()
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // reduce results
    internal::reduce_satisfaction_kernel<<<num_blocks, block_size>>>(
        d_clause_satisfied.data(), num_clauses, d_all_satisfied.data()
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // get result
    d_all_satisfied.copy_to_host(&all_sat, 1);
    
    return all_sat != 0;
}

// utility function to perform unit propagation on GPU
bool unit_propagation_gpu(const vector<int32_t>& clause_lits, const vector<uint32_t>& clause_offsets, unordered_map<uint32_t, bool>& assignment, bool* d_has_conflict) {    
    if (clause_offsets.size() <= 1) {
        return true;  // no clauses
    }
    
    uint32_t num_clauses = clause_offsets.size() - 1;
    const uint32_t MAX_UNIT_LITERALS = 10000;  // maximum number of unit literals per iteration
    
    // allocate GPU memory
    DeviceArray<int32_t> d_clause_lits(clause_lits.size());
    DeviceArray<uint32_t> d_clause_offsets(clause_offsets.size());
    
    // copy clauses to GPU
    d_clause_lits.copy_from_host(clause_lits.data(), clause_lits.size());
    d_clause_offsets.copy_from_host(clause_offsets.data(), clause_offsets.size());
    
    bool changed = true;
    int max_iterations = 1000;  // to prevent infinite loops - dpll solver can loop infinitely
    int iteration = 0;
    
    // main loop for unit propagation
    while (changed && iteration < max_iterations) {
        changed = false;
        iteration++;
        
        vector<uint32_t> vars;
        vector<uint8_t> values;
        
        for (const auto& pair : assignment) {
            vars.push_back(pair.first);
            values.push_back(pair.second ? 1 : 0);
        }
        
        uint32_t num_assignments = vars.size();
        
        // allocate GPU memory for current assignment
        DeviceArray<uint32_t> d_vars(num_assignments > 0 ? num_assignments : 1);
        DeviceArray<uint8_t> d_values(num_assignments > 0 ? num_assignments : 1);
        DeviceArray<int32_t> d_unit_literals(MAX_UNIT_LITERALS);
        DeviceArray<uint32_t> d_num_units(1);
        
        // copy current assignment to GPU
        if (num_assignments > 0) {
            d_vars.copy_from_host(vars.data(), num_assignments);
            d_values.copy_from_host(values.data(), num_assignments);
        }
        
        uint32_t zero = 0;
        d_num_units.copy_from_host(&zero, 1);
        
        // launch kernel to find unit clauses
        int block_size = 256;
        int num_blocks = (num_clauses + block_size - 1) / block_size;
        
        internal::find_unit_clauses_kernel<<<num_blocks, block_size>>>(
            d_clause_lits.data(), d_clause_offsets.data(), num_clauses,
            d_vars.data(), d_values.data(), num_assignments,
            d_unit_literals.data(), d_num_units.data()
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // get number of unit clauses found
        uint32_t num_units = 0;
        d_num_units.copy_to_host(&num_units, 1);
        
        if (num_units == 0) {
            break;  // no more unit clauses
        }
        
        // limit to avoid overflow
        if (num_units > MAX_UNIT_LITERALS) {
            num_units = MAX_UNIT_LITERALS;
        }
        
        // copy unit literals back to host
        vector<int32_t> unit_literals(num_units);
        d_unit_literals.copy_to_host(unit_literals.data(), num_units);
        
        // check for conflicts
        for (int32_t lit : unit_literals) {
            uint32_t var = (lit > 0) ? lit : -lit;
            bool value = (lit > 0);
            
            // check if variable is already assigned
            auto it = assignment.find(var);
            if (it != assignment.end()) {
                if (it->second != value) {
                    // conflict detected
                    if (d_has_conflict != nullptr) {
                        *d_has_conflict = true;
                    }
                    return false;
                }
                // already assigned -> continue
            } else {
                // new assignment
                assignment[var] = value;
                changed = true;
            }
        }
    }
    
    // no conflict detected
    if (d_has_conflict != nullptr) {
        *d_has_conflict = false;
    }
    
    return true;
}

// utility function to launch clause evaluation kernel
void launch_clause_evaluation(const int32_t* d_clause_lits, const uint32_t* d_clause_offsets, uint32_t num_clauses, const uint32_t* d_assignment_vars, const uint8_t* d_assignment_values, uint32_t num_assignments, uint8_t* d_clause_satisfied, uint8_t* d_all_satisfied) {
    // launch kernel to check clauses
    int block_size = 256;
    int num_blocks = (num_clauses + block_size - 1) / block_size;
    
    internal::check_all_clauses_kernel<<<num_blocks, block_size>>>(
        d_clause_lits, d_clause_offsets, num_clauses,
        d_assignment_vars, d_assignment_values, num_assignments,
        d_clause_satisfied
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // reduce results to check if all clauses are satisfied
    internal::reduce_satisfaction_kernel<<<num_blocks, block_size>>>(
        d_clause_satisfied, num_clauses, d_all_satisfied
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace cuda
} // namespace sharpsat
