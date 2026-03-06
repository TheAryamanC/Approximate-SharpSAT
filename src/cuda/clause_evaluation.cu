#ifdef __CUDACC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include "cuda/clause_evaluation.cuh"
#include "cuda/gpu_utils.cuh"
#include <vector>
#include <algorithm>

using namespace std;
namespace sharpsat {
namespace cuda {

namespace internal {

// Kernel to check all clauses in parallel
__global__ void check_all_clauses_kernel(
    const int32_t* clause_lits,
    const uint32_t* clause_offsets,
    uint32_t num_clauses,
    const uint32_t* assignment_vars,
    const uint8_t* assignment_values,
    uint32_t num_assignments,
    uint8_t* clause_satisfied) {
    
    int clause_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (clause_id >= num_clauses) return;
    
    uint32_t start = clause_offsets[clause_id];
    uint32_t end = clause_offsets[clause_id + 1];
    
    bool satisfied = false;
    bool has_unknown = false;
    
    for (uint32_t i = start; i < end; i++) {
        int32_t lit = clause_lits[i];
        uint32_t var = (lit > 0) ? lit : -lit;
        bool sign = (lit > 0);
        
        bool var_assigned = false;
        bool var_value = false;
        
        // Check if variable is assigned
        for (uint32_t j = 0; j < num_assignments; j++) {
            if (assignment_vars[j] == var) {
                var_assigned = true;
                var_value = (assignment_values[j] != 0);
                break;
            }
        }
        
        if (var_assigned) {
            if (var_value == sign) {
                satisfied = true;
                break;
            }
        } else {
            has_unknown = true;
        }
    }
    
    clause_satisfied[clause_id] = (satisfied || has_unknown) ? 1 : 0;
}

// Kernel to find unit clauses
__global__ void find_unit_clauses_kernel(
    const int32_t* clause_lits,
    const uint32_t* clause_offsets,
    uint32_t num_clauses,
    const uint32_t* assignment_vars,
    const uint8_t* assignment_values,
    uint32_t num_assignments,
    int32_t* unit_literals,
    uint32_t* num_units) {
    
    int clause_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (clause_id >= num_clauses) return;
    
    uint32_t start = clause_offsets[clause_id];
    uint32_t end = clause_offsets[clause_id + 1];
    
    uint32_t unassigned_count = 0;
    int32_t unassigned_lit = 0;
    bool has_satisfied = false;
    
    for (uint32_t i = start; i < end; i++) {
        int32_t lit = clause_lits[i];
        uint32_t var = (lit > 0) ? lit : -lit;
        bool sign = (lit > 0);
        
        bool var_assigned = false;
        bool var_value = false;
        
        for (uint32_t j = 0; j < num_assignments; j++) {
            if (assignment_vars[j] == var) {
                var_assigned = true;
                var_value = (assignment_values[j] != 0);
                break;
            }
        }
        
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
    
    if (!has_satisfied && unassigned_count == 1) {
        uint32_t idx = atomicAdd(num_units, 1);
        if (idx < 10000) {  // Safety limit
            unit_literals[idx] = unassigned_lit;
        }
    }
}

}  // namespace internal

// Helper function to convert clauses to GPU format
void convert_clauses_to_gpu_format(
    const vector<vector<int32_t>>& clauses,
    vector<int32_t>& flat_lits,
    vector<uint32_t>& offsets) {
    
    flat_lits.clear();
    offsets.clear();
    
    offsets.push_back(0);
    
    for (const auto& clause : clauses) {
        for (int32_t lit : clause) {
            flat_lits.push_back(lit);
        }
        offsets.push_back(flat_lits.size());
    }
}

// Evaluate clauses on GPU
bool evaluate_clauses_gpu(
    const vector<int32_t>& clause_lits,
    const vector<uint32_t>& clause_offsets,
    const unordered_map<uint32_t, bool>& assignment,
    uint32_t num_variables) {
    
    if (clause_offsets.size() <= 1) {
        return true;  // No clauses
    }
    
    uint32_t num_clauses = clause_offsets.size() - 1;
    
    // Convert assignment to arrays
    vector<uint32_t> vars;
    vector<uint8_t> values;
    
    for (const auto& pair : assignment) {
        vars.push_back(pair.first);
        values.push_back(pair.second ? 1 : 0);
    }
    
    if (vars.empty()) {
        return true;
    }
    
    // Allocate GPU memory
    DeviceArray<int32_t> d_clause_lits(clause_lits.size());
    DeviceArray<uint32_t> d_clause_offsets(clause_offsets.size());
    DeviceArray<uint32_t> d_vars(vars.size());
    DeviceArray<uint8_t> d_values(values.size());
    DeviceArray<uint8_t> d_clause_satisfied(num_clauses);
    
    // Copy to GPU
    d_clause_lits.copy_from_host(clause_lits.data(), clause_lits.size());
    d_clause_offsets.copy_from_host(clause_offsets.data(), clause_offsets.size());
    d_vars.copy_from_host(vars.data(), vars.size());
    d_values.copy_from_host(values.data(), values.size());
    
    // Launch kernel
    int block_size = 256;
    int num_blocks = (num_clauses + block_size - 1) / block_size;
    
    internal::check_all_clauses_kernel<<<num_blocks, block_size>>>(
        d_clause_lits.data(), d_clause_offsets.data(), num_clauses,
        d_vars.data(), d_values.data(), vars.size(),
        d_clause_satisfied.data()
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check results
    vector<uint8_t> clause_results(num_clauses);
    d_clause_satisfied.copy_to_host(clause_results.data(), num_clauses);
    
    for (uint8_t result : clause_results) {
        if (result == 0) {
            return false;  // Found unsatisfied clause
        }
    }
    
    return true;
}

// GPU-accelerated unit propagation
bool unit_propagation_gpu(
    const vector<int32_t>& clause_lits,
    const vector<uint32_t>& clause_offsets,
    unordered_map<uint32_t, bool>& assignment,
    bool* d_has_conflict) {
    
    if (clause_offsets.size() <= 1) {
        return true;
    }
    
    uint32_t num_clauses = clause_offsets.size() - 1;
    const uint32_t MAX_UNIT_LITERALS = 10000;
    
    // Allocate GPU memory
    DeviceArray<int32_t> d_clause_lits(clause_lits.size());
    DeviceArray<uint32_t> d_clause_offsets(clause_offsets.size());
    
    // Copy to GPU
    d_clause_lits.copy_from_host(clause_lits.data(), clause_lits.size());
    d_clause_offsets.copy_from_host(clause_offsets.data(), clause_offsets.size());
    
    // Iteratively find unit clauses and propagate until no more can be found
    bool changed = true;
    int max_iterations = 1000;
    int iteration = 0;
    
    while (changed && iteration < max_iterations) {
        changed = false;
        iteration++;
        
        vector<uint32_t> vars;
        vector<uint8_t> values;
        
        for (const auto& pair : assignment) {
            vars.push_back(pair.first);
            values.push_back(pair.second ? 1 : 0);
        }
        
        // Allocate GPU memory for assignments
        uint32_t num_assignments = vars.size();
        
        DeviceArray<uint32_t> d_vars(num_assignments > 0 ? num_assignments : 1);
        DeviceArray<uint8_t> d_values(num_assignments > 0 ? num_assignments : 1);
        DeviceArray<int32_t> d_unit_literals(MAX_UNIT_LITERALS);
        DeviceArray<uint32_t> d_num_units(1);
        
        if (num_assignments > 0) {
            d_vars.copy_from_host(vars.data(), num_assignments);
            d_values.copy_from_host(values.data(), num_assignments);
        }
        
        uint32_t zero = 0;
        d_num_units.copy_from_host(&zero, 1);
        
        // Launch kernel to find unit clauses
        int block_size = 256;
        int num_blocks = (num_clauses + block_size - 1) / block_size;
        
        internal::find_unit_clauses_kernel<<<num_blocks, block_size>>>(
            d_clause_lits.data(), d_clause_offsets.data(), num_clauses,
            d_vars.data(), d_values.data(), num_assignments,
            d_unit_literals.data(), d_num_units.data()
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy unit literals back to host
        uint32_t num_units = 0;
        d_num_units.copy_to_host(&num_units, 1);
        
        if (num_units == 0) {
            break;
        }
        
        if (num_units > MAX_UNIT_LITERALS) {
            num_units = MAX_UNIT_LITERALS;
        }
        
        // Process unit literals - assign variables and check for conflicts
        vector<int32_t> unit_literals(num_units);
        d_unit_literals.copy_to_host(unit_literals.data(), num_units);
        
        for (int32_t lit : unit_literals) {
            uint32_t var = (lit > 0) ? lit : -lit;
            bool value = (lit > 0);
            
            auto it = assignment.find(var);
            if (it != assignment.end()) {
                if (it->second != value) {
                    if (d_has_conflict != nullptr) {
                        *d_has_conflict = true;
                    }
                    return false;
                }
            } else {
                assignment[var] = value;
                changed = true;
            }
        }
    }
    
    if (d_has_conflict != nullptr) {
        *d_has_conflict = false;
    }
    
    return true;
}

}  // namespace cuda
}  // namespace sharpsat
