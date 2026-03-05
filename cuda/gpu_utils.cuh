#ifndef GPU_UTILS_CUH
#define GPU_UTILS_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <string>

namespace sharpsat {
namespace cuda {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw sharpsat::cuda::CUDAException(__FILE__, __LINE__, error); \
        } \
    } while(0)

// CUDA exception class
class CUDAException : public std::exception {
public:
    CUDAException(const char* file, int line, cudaError_t error);
    const char* what() const noexcept override { return message_.c_str(); }
    
private:
    std::string message_;
};

// GPU device information
struct GPUInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_shared_memory_per_block;
};

// Get GPU information
GPUInfo get_gpu_info(int device_id = 0);

// Print GPU information
void print_gpu_info(int device_id = 0);

// Check if CUDA is available
bool is_cuda_available();

// Get optimal block size for kernel
int get_optimal_block_size(int num_elements);

// Memory management helpers
template<typename T>
class DeviceArray {
public:
    DeviceArray() : data_(nullptr), size_(0) {}
    
    explicit DeviceArray(size_t size) : size_(size) {
        CUDA_CHECK(cudaMalloc(&data_, size * sizeof(T)));
    }
    
    ~DeviceArray() {
        if (data_) {
            cudaFree(data_);
        }
    }
    
    // No copy
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    
    // Move
    DeviceArray(DeviceArray&& other) noexcept 
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this != &other) {
            if (data_) cudaFree(data_);
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    void resize(size_t new_size) {
        if (new_size != size_) {
            if (data_) cudaFree(data_);
            CUDA_CHECK(cudaMalloc(&data_, new_size * sizeof(T)));
            size_ = new_size;
        }
    }
    
    void copy_from_host(const T* host_data, size_t count) {
        CUDA_CHECK(cudaMemcpy(data_, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    void copy_to_host(T* host_data, size_t count) const {
        CUDA_CHECK(cudaMemcpy(host_data, data_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    
    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
    
private:
    T* data_;
    size_t size_;
};

} // namespace cuda
} // namespace sharpsat

#endif // GPU_UTILS_CUH
