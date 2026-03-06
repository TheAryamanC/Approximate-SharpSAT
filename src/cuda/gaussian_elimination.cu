#ifdef __CUDACC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include "cuda/gaussian_elimination.cuh"
#include "cuda/gpu_utils.cuh"
#include "cnf/cnf_structure.h"
#include <vector>
#include <cstring>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace std;
namespace sharpsat {
namespace cuda {

namespace internal {

// Kernel to perform one step in row reduction
// For each row (except pivot row), if it has a 1 in the pivot column, XOR it with the pivot row to eliminate that column
__global__ void row_reduce_step_kernel(uint32_t* matrix, uint8_t* rhs, uint32_t num_rows, uint32_t num_cols, uint32_t pivot_row, uint32_t pivot_col, uint8_t* has_conflict) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= num_rows || row == pivot_row) return; // one thread per row, skip pivot row
    
    // check if this row has a 1 in pivot column
    int word_idx = pivot_col / 32;
    int bit_idx = pivot_col % 32;
    uint32_t val = matrix[row * ((num_cols + 31) / 32) + word_idx];
    
    // if it has a 1 in pivot column, XOR this row with pivot row
    if (val & (1u << bit_idx)) {
        // XOR this row with pivot row
        int num_words = (num_cols + 31) / 32;
        for (int w = 0; w < num_words; w++) {
            int idx = row * num_words + w;
            int pivot_idx = pivot_row * num_words + w;
            atomicXor(&matrix[idx], matrix[pivot_idx]);
        }
        
        // XOR rhs
        uint8_t new_rhs = rhs[row] ^ rhs[pivot_row];
        rhs[row] = new_rhs;
    }
}

// Kernel to find pivot in a given column
// Searches for the first row (starting from start_row) that has a 1 in pivot_col
__global__ void find_pivot_in_column_kernel(const uint32_t* matrix, uint32_t num_rows, uint32_t num_cols, uint32_t pivot_col, uint32_t start_row, uint32_t* pivot_row_out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x + start_row;
    
    if (row >= num_rows) return; // one thread per row starting from current pivot row
    
    // check if this row has a 1 in pivot column
    int word_idx = pivot_col / 32;
    int bit_idx = pivot_col % 32;
    uint32_t val = matrix[row * ((num_cols + 31) / 32) + word_idx];
    
    // if we find a row with a 1 in the pivot column, write its index to pivot_row_out (using atomicMin to get the first one)
    if (val & (1u << bit_idx)) {
        atomicMin(pivot_row_out, row);
    }
}

// Kernel to extract variable assignments from reduced matrix
// For each row, finds the leading variable (first 1) and assigns it the RHS value
__global__ void extract_assignments_kernel(const uint32_t* matrix, const uint8_t* rhs, uint32_t num_rows, uint32_t num_cols, uint32_t* assignment_vars, uint8_t* assignment_values, uint32_t* num_assignments) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= num_rows) return; // one thread per row
    
    // find leading variable (first 1 in row)
    int num_words = (num_cols + 31) / 32;
    int lead_col = -1;
    
    for (int w = 0; w < num_words; w++) {
        uint32_t val = matrix[row * num_words + w];
        if (val != 0) {
            // find first set bit
            for (int b = 0; b < 32; b++) {
                if (val & (1u << b)) {
                    lead_col = w * 32 + b;
                    break;
                }
            }
            if (lead_col >= 0) break;
        }
    }
    
    if (lead_col >= 0 && lead_col < num_cols) {
        // add assignment for this variable
        uint32_t idx = atomicAdd(num_assignments, 1);
        if (idx < num_cols) {
            assignment_vars[idx] = lead_col + 1;  // 1-indexed
            assignment_values[idx] = rhs[row];
        }
    }
}

} // namespace internal

// Utility function to convert XOR constraints to GPU-friendly flattened format
void convert_xors_to_gpu_format(const vector<sharpsat::XorConstraint>& xors, vector<uint32_t>& flat_vars, vector<uint32_t>& offsets, vector<uint8_t>& rhs) {
    // pass by reference to avoid unnecessary copying - so need to clear them first
    flat_vars.clear();
    offsets.clear();
    rhs.clear();
    
    offsets.push_back(0);
    
    // flatten xor constraints into a single array of variables with offsets for each constraint, and a separate array for rhs
    for (const auto& xor_c : xors) {
        for (uint32_t var : xor_c.variables) {
            flat_vars.push_back(var);
        }
        offsets.push_back(flat_vars.size());
        rhs.push_back(xor_c.rhs ? 1 : 0);
    }
}

// main function to perform Gaussian elimination on GPU
bool gaussian_elimination_gpu(const vector<uint32_t>& xor_vars, const vector<uint32_t>& xor_offsets, const vector<uint8_t>& xor_rhs, uint32_t num_variables, unordered_map<uint32_t, bool>& assignment) {
    assignment.clear();
    
    if (xor_offsets.size() <= 1) {
        return true;  // no constraints
    }
    
    uint32_t num_xors = xor_offsets.size() - 1;
    
    // build binary matrix (bit-packed)
    uint32_t num_words = (num_variables + 31) / 32;
    vector<uint32_t> matrix(num_xors * num_words, 0);
    
    for (uint32_t i = 0; i < num_xors; i++) {
        uint32_t start = xor_offsets[i];
        uint32_t end = xor_offsets[i + 1];
        
        for (uint32_t j = start; j < end; j++) {
            uint32_t var = xor_vars[j];
            if (var > 0 && var <= num_variables) {
                uint32_t col = var - 1;
                uint32_t word_idx = col / 32;
                uint32_t bit_idx = col % 32;
                matrix[i * num_words + word_idx] |= (1u << bit_idx);
            }
        }
    }
    
    // allocate GPU memory
    DeviceArray<uint32_t> d_matrix(matrix.size());
    DeviceArray<uint8_t> d_rhs(num_xors);
    DeviceArray<uint8_t> d_has_conflict(1);
    
    // copy data to GPU
    d_matrix.copy_from_host(matrix.data(), matrix.size());
    d_rhs.copy_from_host(xor_rhs.data(), num_xors);
    
    uint8_t has_conflict = 0;
    d_has_conflict.copy_from_host(&has_conflict, 1);
    
    int block_size = 256;
    int num_blocks = (num_xors + block_size - 1) / block_size;
    
    // Gaussian elimination
    uint32_t pivot_row = 0;
    for (uint32_t col = 0; col < num_variables && pivot_row < num_xors; col++) {
        // find pivot
        DeviceArray<uint32_t> d_pivot_row(1);
        uint32_t init_pivot = num_xors;
        d_pivot_row.copy_from_host(&init_pivot, 1);
        
        // launch kernel to find pivot in this column
        internal::find_pivot_in_column_kernel<<<num_blocks, block_size>>>(
            d_matrix.data(), num_xors, num_variables, col, pivot_row, d_pivot_row.data()
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        uint32_t found_pivot;
        d_pivot_row.copy_to_host(&found_pivot, 1);
        
        if (found_pivot >= num_xors) {
            continue;  // no pivot in this column, move to next column
        }
        
        // pivot found - swap rows if needed (done on CPU)
        if (found_pivot != pivot_row) {
            vector<uint32_t> temp_matrix(matrix.size());
            vector<uint8_t> temp_rhs(num_xors);
            d_matrix.copy_to_host(temp_matrix.data(), matrix.size());
            d_rhs.copy_to_host(temp_rhs.data(), num_xors);
            
            // swap
            for (uint32_t w = 0; w < num_words; w++) {
                swap(temp_matrix[pivot_row * num_words + w], temp_matrix[found_pivot * num_words + w]);
            }
            swap(temp_rhs[pivot_row], temp_rhs[found_pivot]);
            
            d_matrix.copy_from_host(temp_matrix.data(), matrix.size());
            d_rhs.copy_from_host(temp_rhs.data(), num_xors);
        }
        
        // launch kernel to eliminate rows below pivot
        internal::row_reduce_step_kernel<<<num_blocks, block_size>>>(
            d_matrix.data(), d_rhs.data(), num_xors, num_variables,
            pivot_row, col, d_has_conflict.data()
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        pivot_row++;
    }
    
    // check for conflicts
    d_has_conflict.copy_to_host(&has_conflict, 1);
    if (has_conflict) {
        return false;
    }
    
    // allocate arrays for back substitution results
    DeviceArray<uint32_t> d_assignment_vars(num_variables);
    DeviceArray<uint8_t> d_assignment_values(num_variables);
    DeviceArray<uint32_t> d_num_assignments(1);
    
    uint32_t zero = 0;
    d_num_assignments.copy_from_host(&zero, 1);
    
    // launch kernel to extract assignments from reduced matrix
    internal::extract_assignments_kernel<<<num_blocks, block_size>>>(
        d_matrix.data(), d_rhs.data(), num_xors, num_variables,
        d_assignment_vars.data(), d_assignment_values.data(), d_num_assignments.data()
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    uint32_t num_assignments;
    d_num_assignments.copy_to_host(&num_assignments, 1);
    
    // fill assignment map
    if (num_assignments > 0) {
        vector<uint32_t> vars(num_assignments);
        vector<uint8_t> values(num_assignments);
        
        d_assignment_vars.copy_to_host(vars.data(), num_assignments);
        d_assignment_values.copy_to_host(values.data(), num_assignments);
        
        for (uint32_t i = 0; i < num_assignments; i++) {
            assignment[vars[i]] = (values[i] != 0);
        }
    }
    
    return true;
}

} // namespace cuda
} // namespace sharpsat
