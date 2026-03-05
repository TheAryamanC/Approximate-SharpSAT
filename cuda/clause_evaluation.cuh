#ifndef CLAUSE_EVALUATION_CUH
#define CLAUSE_EVALUATION_CUH

#include <cstdint>
#include <vector>
#include <unordered_map>

namespace sharpsat {
namespace cuda {

// Represents a clause in GPU-friendly format - a flat array of literals with an offset for each clause
struct GPUClause {
    int32_t* literals;      // Array of literals (positive/negative)
    uint32_t size;          // Number of literals
};

// Evaluate all clauses in parallel on GPU
// Returns true if all clauses are satisfied
bool evaluate_clauses_gpu(
    const std::vector<int32_t>& clause_lits,                // Flattened literals
    const std::vector<uint32_t>& clause_offsets,            // Offset for each clause
    const std::unordered_map<uint32_t, bool>& assignment,   // Variable assignments
    uint32_t num_variables
);

// Check if CNF is satisfied given partial assignment
bool check_cnf_satisfiability_gpu(
    const std::vector<int32_t>& clause_lits,
    const std::vector<uint32_t>& clause_offsets,
    const uint32_t* d_assignment_vars,
    const uint8_t* d_assignment_values,
    uint32_t num_assignments,
    uint32_t num_clauses
);

// Parallel clause evaluation kernel launcher
void launch_clause_evaluation(
    const int32_t* d_clause_lits,
    const uint32_t* d_clause_offsets,
    uint32_t num_clauses,
    const uint32_t* d_assignment_vars,
    const uint8_t* d_assignment_values,
    uint32_t num_assignments,
    uint8_t* d_clause_satisfied,
    uint8_t* d_all_satisfied
);

// Apply unit propagation on GPU
bool unit_propagation_gpu(
    const std::vector<int32_t>& clause_lits,
    const std::vector<uint32_t>& clause_offsets,
    std::unordered_map<uint32_t, bool>& assignment,
    bool* d_has_conflict
);

// Helper: Convert clauses to GPU format
void convert_clauses_to_gpu_format(
    const std::vector<std::vector<int32_t>>& clauses,
    std::vector<int32_t>& flat_lits,
    std::vector<uint32_t>& offsets
);

// Kernel functions
namespace internal {
    // Evaluate a single clause
    __device__ bool evaluate_clause(
        const int32_t* literals,
        uint32_t clause_size,
        const uint32_t* assignment_vars,
        const uint8_t* assignment_values,
        uint32_t num_assignments
    );
    
    // Check all clauses in parallel
    __global__ void check_all_clauses_kernel(
        const int32_t* clause_lits,
        const uint32_t* clause_offsets,
        uint32_t num_clauses,
        const uint32_t* assignment_vars,
        const uint8_t* assignment_values,
        uint32_t num_assignments,
        uint8_t* clause_satisfied
    );
    
    // Reduce clause satisfaction results
    __global__ void reduce_satisfaction_kernel(
        const uint8_t* clause_satisfied,
        uint32_t num_clauses,
        uint8_t* all_satisfied
    );
    
    // Find unit clauses
    __global__ void find_unit_clauses_kernel(
        const int32_t* clause_lits,
        const uint32_t* clause_offsets,
        uint32_t num_clauses,
        const uint32_t* assignment_vars,
        const uint8_t* assignment_values,
        uint32_t num_assignments,
        int32_t* unit_literals,
        uint32_t* num_units
    );
}

} // namespace cuda
} // namespace sharpsat

#endif // CLAUSE_EVALUATION_CUH
