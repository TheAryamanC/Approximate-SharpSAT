#ifdef __CUDACC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include "cuda/gpu_memory_pool.cuh"

namespace sharpsat {
namespace cuda {

// Global memory pool instance
GPUMemoryPool g_gpu_memory_pool;

}  // namespace cuda
}  // namespace sharpsat
