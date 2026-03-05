cmake_minimum_required(VERSION 3.18)

# Use GCC 10 as CUDA host compiler (compatible with CUDA 11.5)
set(CMAKE_C_COMPILER "/usr/bin/gcc-10")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-10")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-10")

project(SharpSAT LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/cuda)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -pthread")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -march=native --expt-relaxed-constexpr --extended-lambda -allow-unsupported-compiler")

# CUDA architectures (adjust based on your GPU)
# CUDA 11.5 supports up to compute_86 (Ampere)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Source files
file(GLOB_RECURSE SOURCES 
    "src/**/*.cpp"
    "src/*.cpp"
)

file(GLOB_RECURSE CUDA_SOURCES 
    "cuda/*.cu"
)

# Create library for CUDA code
add_library(sharp_sat_cuda ${CUDA_SOURCES})
set_target_properties(sharp_sat_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Main executable
add_executable(sharp_sat ${SOURCES})
target_link_libraries(sharp_sat sharp_sat_cuda ${CUDA_LIBRARIES} pthread)

# Test executable (exclude main.cpp from sources)
file(GLOB TEST_SOURCES "tests/*.cpp")
file(GLOB_RECURSE LIB_SOURCES 
    "src/cnf/*.cpp"
    "src/solver/*.cpp"
    "src/utils/*.cpp"
    "src/xor/*.cpp"
)
add_executable(sharp_sat_tests ${TEST_SOURCES} ${LIB_SOURCES})
target_link_libraries(sharp_sat_tests sharp_sat_cuda ${CUDA_LIBRARIES} pthread)

# Enable testing
enable_testing()
add_test(NAME sharp_sat_tests COMMAND sharp_sat_tests)
