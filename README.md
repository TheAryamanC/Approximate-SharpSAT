# SharpSAT: GPU-Accelerated Approximate Model Counter

A high-performance approximate #SAT solver using XOR hashing with machine learning-enhanced hash generation and CUDA parallelization.

## Overview

SharpSAT implements approximate model counting using:
- **XOR-based universal hashing** to reduce solution space
- **Machine learning-enhanced hash generation** for variance reduction
- **CUDA parallelization** for Gaussian elimination, clause evaluation, and unit propagation
- **ApproxMC-inspired architecture** with modern optimizations

## Features

- ✅ Fast CNF parsing (DIMACS format)
- ✅ Sparse XOR constraint generation (traditional + ML-enhanced)
- ✅ GPU-accelerated Gaussian elimination
- ✅ GPU-accelerated clause evaluation and unit propagation
- ✅ DPLL SAT solver with MOMS heuristic
- ✅ Automatic CPU/GPU fallback
- ✅ Memory pooling for efficient GPU buffer reuse
- ✅ Statistical confidence estimation
- ✅ Support for large-scale CNFs (10,000+ variables/clauses)

## Building

### Prerequisites
- CUDA Toolkit 11.8+ (with NVCC compiler)
- g++-10 or newer (as NVCC host compiler)
- C++17 compatible compiler
- Python 3.8+ (for ML components)
- GPU with compute capability 7.5+ (Turing/Ampere/Ada/Hopper)

### Build
```bash
make clean
make
```

The build system automatically:
- Compiles all C++ sources with C++17
- Compiles all CUDA sources with CUDA C++14
- Links CUDA runtime and driver libraries
- Produces `bin/sharp_sat` executable

### Install Python dependencies (for ML hash generation)
```bash
cd ml_model
pip install -r requirements.txt
python train_model.py  # Train the ML model
```

## Usage

### Basic usage
```bash
./bin/sharp_sat <cnf_file>
```

### Example
```bash
./bin/sharp_sat benchmarks/cnfs/easy_500_0.cnf
./bin/sharp_sat benchmarks/cnfs/horn_1000_0.cnf
```

The counter automatically:
- Detects CUDA availability and enables GPU acceleration
- Uses ML-enhanced hash generation if model is available
- Configures epsilon=0.8 and delta=0.2 for approximate counting
- Logs progress and results to console

## Project Structure

```
approx_model_counting/
├── Makefile                     # Build system
├── README.md                    # This file
├── GPU_INTEGRATION.md           # GPU acceleration documentation
├── benchmarks/                  # Benchmark CNF files and scripts
│   ├── cnfs/                    # Test CNF files (easy, horn, random, etc.)
│   ├── generate_cnfs.py         # CNF generator script
│   └── run_benchmark_trials.py  # Benchmark runner
├── bin/                         # Build output directory
│   └── sharp_sat                # Main executable
├── cuda/                        # CUDA implementation files
│   ├── clause_evaluation.cu     # GPU clause evaluation & unit propagation
│   ├── gaussian_elimination.cu  # GPU Gaussian elimination
│   ├── gpu_memory_pool.cu       # GPU memory pool for buffer reuse
│   └── gpu_utils.cu             # CUDA utilities and error checking
├── include/                     # Header files
│   ├── cuda_interface.h         # CUDA availability checking
│   ├── cnf/
│   │   ├── cnf_parser.h         # DIMACS CNF parser
│   │   └── cnf_structure.h      # CNF data structures
│   ├── cuda/
│   │   ├── clause_evaluation.cuh    # GPU clause evaluation headers
│   │   ├── gaussian_elimination.cuh # GPU Gaussian elimination headers
│   │   ├── gpu_memory_pool.cuh      # GPU memory pool headers
│   │   └── gpu_utils.cuh            # GPU utilities headers
│   ├── solver/
│   │   ├── approximate_counter.h    # Main ApproxMC implementation
│   │   ├── cnf_simplifier.h         # CNF simplification (unit prop, pure literal)
│   │   └── sat_solver.h             # DPLL SAT solver with GPU support
│   ├── utils/
│   │   ├── logger.h             # Logging system
│   │   └── timer.h              # Performance timing
│   └── xor/
│       ├── ml_hash_interface.h  # ML model interface
│       └── xor_hash_generator.h # XOR constraint generation
├── ml_model/                    # Machine learning components
│   ├── hash_predictor.py        # ML model for hash prediction
│   ├── ml_server.py             # ML inference server
│   ├── train_model.py           # Model training script
│   ├── requirements.txt         # Python dependencies
│   └── training_cnfs/           # Training data
├── src/                         # C++ source files
│   ├── main.cpp                 # Entry point
│   ├── cnf/
│   │   ├── cnf_parser.cpp       # CNF parser implementation
│   │   └── cnf_structure.cpp    # CNF structure implementation
│   ├── solver/
│   │   ├── approximate_counter.cpp  # ApproxMC implementation
│   │   ├── cnf_simplifier.cpp       # CNF simplification implementation
│   │   └── sat_solver.cpp           # DPLL solver implementation
│   ├── utils/
│   │   └── timer.cpp            # Timer implementation
│   └── xor/
│       ├── ml_hash_interface.cpp    # ML interface implementation
│       └── xor_hash_generator.cpp   # XOR generation implementation
└── tests/                       # Test files
    ├── test_*.cpp               # Individual test modules
    └── test_runner.cpp          # Test harness
```

## How It Works

### ApproxMC Algorithm (Implemented)
1. **Pivot Threshold Finding**: 
   - Performs binary search to find optimal number of XOR constraints
   - Uses 3 independent samples per threshold value
   - Target: Find threshold where ≥ 50% of SAT checks succeed
   
2. **XOR Constraint Generation**:
   - Generates random XOR constraints (traditional method)
   - Optionally uses ML model to predict better constraints (ML-enhanced)
   - Each constraint partitions solution space by half
   
3. **Gaussian Elimination** (GPU-accelerated):
   - Reduces XOR constraint system to row echelon form
   - Runs in parallel on GPU for large constraint sets
   - Extracts variable assignments from reduced system
   
4. **CNF Simplification**:
   - Applies XOR-derived assignments to original CNF
   - Performs unit propagation (GPU-accelerated when enabled)
   - Performs pure literal elimination
   
5. **SAT Solving** (DPLL with GPU support):
   - Checks satisfiability of simplified formula
   - Uses MOMS heuristic for variable selection
   - GPU-accelerated clause evaluation and unit propagation
   - Falls back to CPU for small formulas
   
6. **Counting & Statistical Estimation**:
   - Runs multiple iterations at pivot threshold
   - Estimates total model count: count = 2^(pivot) × (solutions / iterations)
   - Computes confidence bounds based on epsilon/delta parameters

### GPU Acceleration Details

**What runs on GPU:**
- Gaussian elimination of XOR constraints
- Clause evaluation (checking all clauses in parallel)
- Unit propagation (finding unit clauses in parallel)
- Memory pooling for efficient buffer reuse

**Automatic CPU/GPU switching:**
- System checks CUDA availability at startup
- Enables GPU acceleration if CUDA detected
- Falls back to CPU for small problems or if GPU unavailable
- No user configuration required

**Performance characteristics:**
- GPU wins on large formulas (1000+ variables, 2500+ clauses)
- CPU wins on small formulas (<500 variables due to transfer overhead)
- Typical speedup: 2-5x on large easy-SAT and Horn formulas

See [GPU_INTEGRATION.md](GPU_INTEGRATION.md) for detailed GPU documentation.

## Performance

### Tested CNF Files

The implementation has been tested on various CNF types:
- **Standard**: Random 3-SAT formulas (50-500 variables)
- **Easy-SAT**: Satisfiable formulas with many solutions (500-3000 variables)
- **Horn**: Polynomial-time solvable formulas (1000-4000 variables)
- **Random**: General random formulas (500-3000 variables)
- **Large**: High-complexity formulas (1000-4000 variables)
- **K-SAT**: Variable clause length formulas (500-3000 variables, k=4-7)

### GPU Performance Notes

**GPU acceleration is beneficial for:**
- Large formulas (1000+ variables, 2500+ clauses)
- Many SAT solver iterations (pivot finding + multiple iterations)
- XOR constraint systems with many constraints

**CPU is faster for:**
- Small formulas (<500 variables)
- Quick SAT checks (GPU transfer overhead dominates)

The system automatically selects the best approach based on CUDA availability.

### Example Timings

Example runs on typical hardware (RTX 3080, i7-10700K):
```
standard_50_0.cnf    (50 vars, 213 clauses)   : ~0.2s
easy_500_0.cnf       (500 vars, 1000 clauses) : ~1.5s (10 iterations)
horn_1000_0.cnf      (1000 vars, 2500 clauses): ~3.3s (10 iterations)
random_2000_2.cnf    (2000 vars, 4000 clauses): ~10-15s
```

Actual performance depends on:
- Formula structure (easy-SAT vs. hard-SAT)
- Pivot threshold (number of XOR constraints)
- Number of iterations required
- GPU compute capability

## Documentation

### Complete Documentation Files

- **[README.md](README.md)** - This file: Quick start guide and overview
- **[CODEBASE_REFERENCE.txt](CODEBASE_REFERENCE.txt)** - Complete reference documenting every file and function in the codebase, explaining what each does, why it exists, and how it connects to the approximate counting algorithm
- **[GPU_INTEGRATION.md](GPU_INTEGRATION.md)** - Detailed GPU acceleration documentation including architecture, usage, and performance characteristics

### Key Documentation Sections

The CODEBASE_REFERENCE.txt provides:
- Function-by-function walkthrough of entire codebase
- Detailed algorithm explanations (ApproxMC, DPLL, Gaussian elimination)
- Complete execution flow from main() to results
- GPU acceleration integration details
- Performance characteristics and complexity analysis
- Testing and debugging information

## References

- [ApproxMC](https://github.com/meelgroup/approxmc) - Original approximate counter
- [Sparse XOR Hashing](https://arxiv.org/abs/2004.14692)
- [GPUSAT](https://github.com/daajoe/GPUSAT) - GPU SAT solver reference

## Author

Aryaman Chawla - aryamanchawla2025@u.northwestern.edu
