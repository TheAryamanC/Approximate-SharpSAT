#ifndef CUDA_INTERFACE_H
#define CUDA_INTERFACE_H

#include <cstdint>
#include <vector>
#include <unordered_map>

/// <summary>
///  C++ interface to CUDA functions (implementation in CUDA files)
/// </summary>

// Forward declaration from cnf_structure.h
namespace sharpsat {
    struct XorConstraint;
}

namespace sharpsat {
namespace cuda {

// Print GPU information
void print_gpu_info(int device_id = 0);

// Check if CUDA is available
bool is_cuda_available();

// Gaussian elimination on GPU
bool gaussian_elimination_gpu(
    const std::vector<uint32_t>& xor_vars,
    const std::vector<uint32_t>& xor_offsets,
    const std::vector<uint8_t>& xor_rhs,
    uint32_t num_variables,
    std::unordered_map<uint32_t, bool>& assignment
);

// Helper: Convert XOR constraints to GPU format
void convert_xors_to_gpu_format(
    const std::vector<sharpsat::XorConstraint>& xors,
    std::vector<uint32_t>& flat_vars,
    std::vector<uint32_t>& offsets,
    std::vector<uint8_t>& rhs
);

}  // namespace cuda
}  // namespace sharpsat

#endif  // CUDA_INTERFACE_H
