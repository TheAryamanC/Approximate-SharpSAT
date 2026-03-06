#ifndef GAUSSIAN_ELIMINATION_CUH
#define GAUSSIAN_ELIMINATION_CUH

#include <cstdint>
#include <vector>
#include <unordered_map>

namespace sharpsat {
    struct XorConstraint;
}

namespace sharpsat {
namespace cuda {

// Gaussian elimination (GPU) - converts XOR constraints into partial variable assignments
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

// Internal GPU kernels (implemented in gaussian_elimination.cu)
namespace internal {
    // Perform row reduction step on a binary matrix
    // Eliminates pivot column in all rows except pivot row
    __global__ void row_reduce_step_kernel(
        uint32_t* matrix,
        uint8_t* rhs,
        uint32_t num_rows,
        uint32_t num_cols,
        uint32_t pivot_row,
        uint32_t pivot_col,
        uint8_t* has_conflict
    );
    
    // Find pivot row for given column (starting from start_row)
    __global__ void find_pivot_in_column_kernel(
        const uint32_t* matrix,
        uint32_t num_rows,
        uint32_t num_cols,
        uint32_t pivot_col,
        uint32_t start_row,
        uint32_t* pivot_row_out
    );
    
    // Extract variable assignments from reduced matrix
    __global__ void extract_assignments_kernel(
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
