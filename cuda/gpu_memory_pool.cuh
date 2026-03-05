#ifndef GPU_MEMORY_POOL_CUH
#define GPU_MEMORY_POOL_CUH

#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include "gpu_utils.cuh"

namespace sharpsat {
namespace cuda {

// GPU Memory Pool - reduces allocation overhead by reusing GPU buffers (for many small CUDA allocations)
class GPUMemoryPool {
public:
    GPUMemoryPool() = default;
    ~GPUMemoryPool() {
        // Free all cached buffers
        for (auto& pair : buffers_) {
            if (pair.second.ptr) {
                cudaFree(pair.second.ptr);
            }
        }
    }
    
    // Get buffer of at least min_size bytes, reusing if available
    void* get_buffer(size_t min_size, const char* name = "buffer") {
        auto it = buffers_.find(name);
        
        if (it != buffers_.end() && it->second.size >= min_size) {
            // Reuse existing buffer
            return it->second.ptr;
        }
        
        // Need to allocate new buffer
        void* ptr = nullptr;
        if (it != buffers_.end() && it->second.ptr) {
            cudaFree(it->second.ptr);
        }
        
        CUDA_CHECK(cudaMalloc(&ptr, min_size));
        buffers_[name] = {ptr, min_size};
        return ptr;
    }
    
    // Clear all cached buffers
    void clear() {
        for (auto& pair : buffers_) {
            if (pair.second.ptr) {
                cudaFree(pair.second.ptr);
            }
        }
        buffers_.clear();
    }
    
    // Get statistics
    size_t get_total_allocated() const {
        size_t total = 0;
        for (const auto& pair : buffers_) {
            total += pair.second.size;
        }
        return total;
    }
    
private:
    struct Buffer {
        void* ptr;
        size_t size;
    };
    
    std::unordered_map<std::string, Buffer> buffers_;
};

// Global memory pool instance
extern GPUMemoryPool g_gpu_memory_pool;

}  // namespace cuda
}  // namespace sharpsat

#endif  // GPU_MEMORY_POOL_CUH
