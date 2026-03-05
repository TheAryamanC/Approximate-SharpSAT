#ifndef CLAUSE_EVALUATION_CUH
#define CLAUSE_EVALUATION_CUH

#include <cstdint>
#include <vector>
#include <unordered_map>

namespace sharpsat {
namespace cuda {

// Evaluate all clauses in parallel on GPU
// Returns true if all clauses are satisfied or can be satisfied
bool evaluate_clauses_gpu(
    const std::vector<int32_t>& clause_lits,                // Flattened literals
    const std::vector<uint32_t>& clause_offsets,            // Offset for each clause
    const std::unordered_map<uint32_t, bool>& assignment,   // Variable assignments
    uint32_t num_variables
);

// Apply unit propagation on GPU
// Returns true if successful (no conflicts), false if conflict detected
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

}  // namespace cuda
}  // namespace sharpsat

#endif  // CLAUSE_EVALUATION_CUH
