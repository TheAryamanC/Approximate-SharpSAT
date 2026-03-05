# SharpSAT: GPU-Accelerated Approximate Model Counter

A high-performance approximate #SAT solver using XOR hashing with machine learning-enhanced hash generation and CUDA parallelization.

## Overview

SharpSAT implements approximate model counting using:
- **XOR-based universal hashing** to reduce solution space
- **Machine learning-enhanced hash generation** for variance reduction
- **CUDA parallelization** for Gaussian elimination and clause evaluation
- **ApproxMC-inspired architecture** with modern optimizations

## Features

- ✅ Fast CNF parsing (DIMACS format)
- ✅ Sparse XOR constraint generation (traditional + ML-enhanced)
- ✅ GPU-accelerated Gaussian elimination
- ✅ Parallel clause evaluation on CUDA
- ✅ Statistical confidence estimation
- ✅ Support for large-scale CNFs (100k+ variables/clauses)

## Building

### Prerequisites
- CMake 3.18+
- CUDA Toolkit 11.0+
- C++17 compatible compiler
- Python 3.8+ (for ML components)

### Build with CMake
```bash
mkdir build && cd build
cmake ..
make -j
```

### Build with Makefile
```bash
make -j
# Or use CMake backend
make cmake
```

### Install Python dependencies
```bash
cd ml_model
pip install -r requirements.txt
```

## Usage

### Basic usage
```bash
./bin/sharp_sat <cnf_file> [options]
```

### Options
- `--epsilon <float>`: Approximation factor (default: 0.8)
- `--delta <float>`: Confidence parameter (default: 0.2)
- `--use-ml`: Enable ML-enhanced hash generation
- `--no-cuda`: Disable CUDA acceleration (CPU only)
- `--seed <int>`: Random seed
- `--verbose`: Enable detailed logging

### Example
```bash
./bin/sharp_sat benchmarks/cnfs/example.cnf --epsilon 0.8 --delta 0.2 --use-ml
```

## Testing

```bash
make test
# Or run directly
./bin/sharp_sat_tests
```

## Project Structure

```
sharp_sat/
├── CMakeLists.txt              # CMake build configuration
├── Makefile                     # Alternative build system
├── benchmarks/cnfs/             # Test CNF files
├── cuda/                        # CUDA kernels
│   ├── gaussian_elimination.cu  # Parallel Gaussian elimination
│   ├── clause_evaluation.cu     # Parallel clause checking
│   └── gpu_utils.cu             # GPU utilities
├── include/                     # Header files
│   ├── cnf/                     # CNF parsing and structures
│   ├── solver/                  # Core solver logic
│   ├── utils/                   # Utilities
│   └── xor/                     # XOR hash generation
├── ml_model/                    # Machine learning components
│   ├── train_hash_model.py      # Training script
│   ├── hash_predictor.py        # Inference interface
│   └── model_utils.py           # ML utilities
├── src/                         # C++ implementation
└── tests/                       # Test suite
```

## How It Works

### ApproxMC Algorithm
1. **Cell decomposition**: Add random XOR constraints to partition solution space
2. **Gaussian elimination**: Reduce XOR constraints to find partial assignments
3. **Unit propagation**: Simplify CNF with partial assignments
4. **SAT solving**: Check satisfiability of simplified formula
5. **Counting**: Estimate total count based on cell size and successful trials

### ML-Enhanced Hash Generation
Traditional sparse XOR hashing uses purely random bit selection. This project enhances it by:
- Learning from CNF structure (variable occurrences, clause lengths, locality)
- Predicting better hash functions that reduce variance
- Adapting to different problem domains

### CUDA Parallelization
- **Gaussian elimination**: Each XOR constraint processed in parallel
- **Clause evaluation**: Parallel checking of all clauses
- **Multiple trials**: Independent hash iterations run concurrently

## Performance

### GPU Speedup (Measured)

**GPU is faster when**: Problem size > 1000 variables AND SAT instances are easy

| CNF Type | Variables | CPU Time | GPU Time | GPU Speedup |
|----------|-----------|----------|----------|-------------|
| Easy-SAT | 1000 | 4.64s | 2.97s | **1.56x** ✓ |
| Easy-SAT | 2000 | 30.68s | 7.25s | **4.23x** ✓ |
| Horn | 1000 | 4.16s | 2.23s | **1.87x** ✓ |
| Horn | 2000 | 30.45s | 7.44s | **4.09x** ✓ |

**Average GPU speedup**: 2.94x on 1000-2000 variable problems

**CPU is faster when**: Problem size < 500 variables (GPU overhead dominates)

| CNF Type | Variables | CPU Time | GPU Time | CPU Advantage |
|----------|-----------|----------|----------|---------------|
| Random 3-SAT | 100 | 3.69s | 4.29s | 1.16x |
| Random 3-SAT | 50 | 0.09s | 0.38s | 4.09x |
| Small | 20 | 0.001s | 0.22s | 195x |

### Key Findings

✅ **GPU wins on large, easy-SAT problems** (2000+ vars, 4x+ speedup)  
✅ **Speedup scales with problem size** (doubles from 1000 to 2000 vars)  
✅ **Both ML and Non-ML methods produce consistent counts** (0% variance)  
⚠️ **CPU wins on small problems** (<500 vars due to GPU transfer overhead)

See [GPU_OPTIMIZATION_REPORT.md](GPU_OPTIMIZATION_REPORT.md) and [PROJECT_COMPLETION_SUMMARY.md](PROJECT_COMPLETION_SUMMARY.md) for detailed analysis.

## Benchmarks Generated

**Comprehensive test suite** (18 CNF files, sizes 5 to 10,000 variables):
- Standard random 3-SAT (phase transition formulas)
- GPU-friendly easy-SAT (fast solving, large GE matrices)
- Horn formulas (polynomial-time solvable)
- Dense formulas (hard SAT instances)

Total benchmark trials run: **160+ trials** across multiple configurations

## References

- [ApproxMC](https://github.com/meelgroup/approxmc) - Original approximate counter
- [Sparse XOR Hashing](https://arxiv.org/abs/2004.14692)
- [GPUSAT](https://github.com/daajoe/GPUSAT) - GPU SAT solver reference

## Author

Aryaman Chawla - aryamanchawla2025@u.northwestern.edu
