#ifdef __CUDACC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include "cuda/gpu_utils.cuh"
#include <iostream>
#include <sstream>

using namespace std;
namespace sharpsat {
namespace cuda {

// function to return error string for a given CUDA error code
CUDAException::CUDAException(const char* file, int line, cudaError_t error) {
    ostringstream oss;
    oss << "CUDA error at " << file << ":" << line << ": " << cudaGetErrorString(error) << " (" << error << ")";
    message_ = oss.str();
}

// get GPU information
GPUInfo get_gpu_info(int device_id) {
    GPUInfo info;
    info.device_id = device_id;
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    info.name = prop.name;
    info.total_memory = prop.totalGlobalMem;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.multiprocessor_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.max_shared_memory_per_block = prop.sharedMemPerBlock;
    
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    info.free_memory = free_mem;
    
    return info;
}

// print GPU information
void print_gpu_info(int device_id) {
    try {
        GPUInfo info = get_gpu_info(device_id);
        
        cout << "GPU Information:" << endl;
        cout << "  Device ID: " << info.device_id << endl;
        cout << "  Name: " << info.name << endl;
        cout << "  Compute Capability: " << info.compute_capability_major << "." << info.compute_capability_minor << endl;
        cout << "  Total Memory: " << info.total_memory / (1024*1024) << " MB" << endl;
        cout << "  Free Memory: " << info.free_memory / (1024*1024) << " MB" << endl;
        cout << "  Multiprocessors: " << info.multiprocessor_count << endl;
        cout << "  Max Threads per Block: " << info.max_threads_per_block << endl;
        cout << "  Max Shared Memory per Block: " << info.max_shared_memory_per_block / 1024 << " KB" << endl;
    } catch (const CUDAException& e) {
        cerr << "Error getting GPU info: " << e.what() << endl;
    }
}

// check if CUDA is available
bool is_cuda_available() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess) {
        return false;
    }
    
    return device_count > 0;
}

// Initialize CUDA context (must be called before any CUDA operations, especially in multi-threaded context)
bool initialize_cuda_context(int device_id) {
    try {
        // Explicitly set the device
        CUDA_CHECK(cudaSetDevice(device_id));
        
        // Force context initialization by doing a simple operation
        CUDA_CHECK(cudaFree(0));
        
        // Synchronize to ensure context is fully initialized
        CUDA_CHECK(cudaDeviceSynchronize());
        
        return true;
    } catch (const CUDAException& e) {
        cerr << "Failed to initialize CUDA context: " << e.what() << endl;
        return false;
    }
}

// get optimal block size for kernel launches
int get_optimal_block_size(int num_elements) {
    if (num_elements <= 0) return 32;
    
    // Common block sizes: 32, 64, 128, 256, 512, 1024
    if (num_elements < 64) return 32;
    if (num_elements < 128) return 64;
    if (num_elements < 256) return 128;
    if (num_elements < 512) return 256;
    if (num_elements < 1024) return 512;
    return 1024;
}

} // namespace cuda
} // namespace sharpsat
