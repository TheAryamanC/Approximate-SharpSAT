# SharpSAT: GPU-Accelerated Approximate Model Counter

This is an approximate #SAT solver using XOR hashing. I have added machine-learning enhanced XOR hash generation and parallelization using CUDA capcbilities.

## To run the program

### Software needed
- CUDA Toolkit 11.8+ (with NVCC compiler)
- g++-10 or newer (as NVCC host compiler)
- C++17 compatible compiler
- Python 3.8+ (for ML components)
- GPU with compute capability 7.5+ (Turing/Ampere/Ada/Hopper)

### Build
```bash
cd src/ml_model
pip install -r requirements.txt
python train_model.py
cd ..
make
```

This will train the model, and produce an executable sharp_sat file that you can run/

## Usage

```bash
./bin/sharp_sat <cnf_file>
```

The counter automatically:
- Detects CUDA availability and enables GPU acceleration
- Uses ML-enhanced hash generation if model is available
- Configures epsilon=0.8 and delta=0.2 for approximate counting
- Logs progress and results to console

## How It Works

1. **Pivot Threshold Finding**: 
   - Performs binary search to find optimal number of XOR constraints
   - The goal is to find a threshold where over 50% of SAT checks succeed
   
2. **XOR Constraint Generation**:
   - Generates random XOR constraints - with the option of using an XGBoost model to predict better constraints
   - Each constraint partitions the solution space by half
   
3. **Gaussian Elimination**:
   - Reduces XOR constraint system to row echelon form
   - Extracts variable assignments from reduced system
   
4. **CNF Simplification**:
   - Applies XOR-derived assignments to simplify the original CNF
   
5. **SAT Solving**:
   - Checks satisfiability of simplified formula using DPLL
   - Uses the MOMS heuristic for variable selection
   
6. **Counting & Statistical Estimation**:
   - Runs multiple iterations at pivot threshold
   - Estimates total model count: count = 2^(pivot) * (solutions / iterations)
   - Computes confidence bounds based on epsilon/delta parameters

## Performance

The program has been tested on various CNF types:
- **Standard**: Random 3-SAT formulas (50-500 variables)
- **Easy-SAT**: Satisfiable formulas with many solutions (500-3000 variables)
- **Horn**: Polynomial-time solvable formulas (1000-4000 variables)
- **Random**: General random formulas (500-3000 variables)
- **Large**: High-complexity formulas (1000-4000 variables)
- **K-SAT**: Variable clause length formulas (500-3000 variables, k=4-7)

### Results

Using GPUs is beneficial for large formulas, but the overhead from using GPUs makes it unsuitable for small formulas