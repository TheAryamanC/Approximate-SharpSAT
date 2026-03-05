#ifndef GAUSSIAN_ELIMINATION_CUH
#define GAUSSIAN_ELIMINATION_CUH

#include <cstdint>
#include <vector>
#include <unordered_map>

// Forward declare in sharpsat namespace
namespace sharpsat {
    struct XorConstraint;
}

namespace sharpsat {
namespace cuda {

// Represents XOR constraint in GPU-friendly format
struct GPUXorConstraint {
    uint32_t* variables;    // Array of variables
    uint32_t size;          // Number of variables
    bool rhs;               // Right-hand side
};

// Gaussian elimination on GPU
// Converts XOR constraints into partial variable assignments
// Returns true if successful, false if contradiction detected
bool gaussian_elimination_gpu(
    const std::vector<uint32_t>& xor_vars,          // Flattened variable array
    const std::vector<uint32_t>& xor_offsets,       // Offset for each XOR constraint
    const std::vector<uint8_t>& xor_rhs,            // RHS for each constraint (0 or 1)
    uint32_t num_variables,                         // Total number of variables
    std::unordered_map<uint32_t, bool>& assignment  // Output: variable assignments
);

// Helper: Convert XOR constraints to GPU format
void convert_xors_to_gpu_format(
    const std::vector<sharpsat::XorConstraint>& xors,
    std::vector<uint32_t>& flat_vars,
    std::vector<uint32_t>& offsets,
    std::vector<uint8_t>& rhs
);

// Matrix row reduction on GPU (used internally)
namespace internal {
    // Perform row reduction on a binary matrix
    __global__ void row_reduce_kernel(
        uint32_t* matrix,
        uint8_t* rhs,
        uint32_t num_rows,
        uint32_t num_cols,
        uint32_t pivot_col,
        uint8_t* has_conflict
    );
    
    // Find pivot row for given column
    __global__ void find_pivot_kernel(
        const uint32_t* matrix,
        uint32_t num_rows,
        uint32_t num_cols,
        uint32_t col,
        uint32_t* pivot_row
    );
    
    // Back substitution kernel
    __global__ void back_substitution_kernel(
        const uint32_t* matrix,
        const uint8_t* rhs,
        uint32_t num_rows,
        uint32_t num_cols,
        uint32_t* assignment_vars,
        uint8_t* assignment_values,
        uint32_t* num_assignments
    );
}

} // namespace cuda
} // namespace sharpsat

#endif // GAUSSIAN_ELIMINATION_CUH
