import random
import argparse
from pathlib import Path
from typing import List


def generate_3sat(num_vars: int, num_clauses: int, output_file: str, seed: int = 42) -> None:
    """Generate a random 3-SAT instance"""
    random.seed(seed)
    
    with open(output_file, 'w') as f:
        f.write(f"c Random 3-SAT formula for benchmarking\n")
        f.write(f"c Generated with seed {seed}\n")
        f.write(f"p cnf {num_vars} {num_clauses}\n")
        
        for i in range(num_clauses):
            # pick 3 random distinct variables
            vars_in_clause = random.sample(range(1, num_vars + 1), 3)
            # randomly negate each variable
            clause = [v * random.choice([1, -1]) for v in vars_in_clause]
            f.write(' '.join(map(str, clause)) + ' 0\n')


def generate_large_3sat(num_vars: int, clause_to_var_ratio: float, output_file: str) -> None:
    """Generate a large random 3-SAT instance with specified clause-to-variable ratio"""
    num_clauses = int(num_vars * clause_to_var_ratio)
    
    clauses = []
    for _ in range(num_clauses):
        # generate 3 random distinct variables
        vars_in_clause = random.sample(range(1, num_vars + 1), 3)
        
        # randomly negate each variable
        literals = [v if random.random() > 0.5 else -v for v in vars_in_clause]
        clauses.append(literals)
        
    with open(output_file, 'w') as f:
        f.write(f"c Large 3-SAT formula for GPU testing\n")
        f.write(f"c {num_vars} variables, {num_clauses} clauses (ratio: {clause_to_var_ratio})\n")
        f.write(f"p cnf {num_vars} {num_clauses}\n")
        
        for clause in clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")


def generate_easy_sat_cnf(num_vars: int, output_file: str) -> None:
    """Generate a CNF that is easy to satisfy but has many variables"""
    # fewer clauses with lower ratio makes SAT easier
    num_clauses = int(num_vars * 2.0)
    
    clauses = []
    
    # add some positive unit clauses to make it trivially satisfiable
    for i in range(min(10, num_vars)):
        clauses.append([i + 1])
    
    # add larger clauses which are easier to satisfy
    for _ in range(num_clauses - 10):
        clause_size = random.randint(4, 6)
        vars_in_clause = random.sample(range(1, num_vars + 1), min(clause_size, num_vars))
        literals = [v if random.random() > 0.5 else -v for v in vars_in_clause]
        clauses.append(literals)
        
    with open(output_file, 'w') as f:
        f.write(f"c Easy-SAT formula for GPU Gaussian elimination testing\n")
        f.write(f"c {num_vars} variables, {len(clauses)} clauses\n")
        f.write(f"c Designed to be easy to satisfy (fast SAT) but large (slow GE)\n")
        f.write(f"p cnf {num_vars} {len(clauses)}\n")
        
        for clause in clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")


def generate_horn_cnf(num_vars: int, output_file: str) -> None:
    """Generate Horn CNF (polynomial-time solvable)"""
    num_clauses = int(num_vars * 2.5)
    
    clauses = []
    
    # horn clauses have at most one positive literal
    for _ in range(num_clauses):
        clause_size = random.randint(2, 4)
        vars_selected = random.sample(range(1, num_vars + 1), clause_size)
        
        has_positive = random.random() > 0.3
        literals = []
        for i, v in enumerate(vars_selected):
            if has_positive and i == 0:
                literals.append(v)  # one positive
            else:
                literals.append(-v)  # rest negative
        clauses.append(literals)
        
    with open(output_file, 'w') as f:
        f.write(f"c Horn formula (polynomial-time solvable)\n")
        f.write(f"c {num_vars} variables, {len(clauses)} clauses\n")
        f.write(f"p cnf {num_vars} {len(clauses)}\n")
        
        for clause in clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")


def generate_ksat_cnf(num_vars: int, k: int, output_file: str) -> None:
    """Generate k-SAT CNF with exactly k literals per clause"""
    num_clauses = int(num_vars * 4.0)  # Standard ratio
    
    clauses = []
    k_actual = min(k, num_vars)  # Can't have more literals than variables
    
    for _ in range(num_clauses):
        vars_selected = random.sample(range(1, num_vars + 1), k_actual)
        literals = [v if random.random() > 0.5 else -v for v in vars_selected]
        clauses.append(literals)
        
    with open(output_file, 'w') as f:
        f.write(f"c {k}-SAT formula\n")
        f.write(f"c {num_vars} variables, {len(clauses)} clauses, {k_actual} literals per clause\n")
        f.write(f"p cnf {num_vars} {len(clauses)}\n")
        
        for clause in clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")


def generate_random_cnf(num_vars: int, output_file: str) -> None:
    """Generate random CNF with variable-sized clauses"""
    num_clauses = int(num_vars * 3.5)  # Moderate ratio
    
    clauses = []
    
    for _ in range(num_clauses):
        # Random clause size between 2 and 7
        clause_size = random.randint(2, min(7, num_vars))
        vars_selected = random.sample(range(1, num_vars + 1), clause_size)
        literals = [v if random.random() > 0.5 else -v for v in vars_selected]
        clauses.append(literals)
        
    with open(output_file, 'w') as f:
        f.write(f"c Random CNF with variable clause sizes (2-7)\n")
        f.write(f"c {num_vars} variables, {len(clauses)} clauses\n")
        f.write(f"p cnf {num_vars} {len(clauses)}\n")
        
        for clause in clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")


def generate_all(cnf_dir: Path, standard_count: int = 4, large_count: int = 4, dense_count: int = 3, easy_count: int = 4, horn_count: int = 4, ksat_count: int = 4, random_count: int = 4) -> None:
    """Generate all types of CNF files with configurable counts (max 5000 variables)"""
    cnf_dir.mkdir(exist_ok=True)
    
    base_seed = 42
    total_generated = 0
    
    # Maximum variables allowed
    MAX_VARS = 5000
    
    # 1. Standard 3-SAT instances (cap at 5000)
    if standard_count > 0:
        standard_sizes = [50, 100, 200, 500, 1000, 2000, 5000] * (standard_count // 7 + 1)
        standard_sizes = standard_sizes[:standard_count]
        for i, num_vars in enumerate(standard_sizes):
            num_vars = min(num_vars, MAX_VARS)
            num_clauses = int(num_vars * 4.26)  # Standard ratio
            output = cnf_dir / f"standard_{num_vars}_{i}.cnf"
            generate_3sat(num_vars, num_clauses, str(output), base_seed + i)
            total_generated += 1
    
    # 2. Large 3-SAT instances (phase transition, cap at 5000)
    if large_count > 0:
        large_sizes = [1000, 2000, 3000, 4000, 5000] * (large_count // 5 + 1)
        large_sizes = large_sizes[:large_count]
        for i, size in enumerate(large_sizes):
            size = min(size, MAX_VARS)
            output = cnf_dir / f"large_{size}_{i}.cnf"
            generate_large_3sat(size, clause_to_var_ratio=4.3, output_file=str(output))
            total_generated += 1
    
    # 3. Dense formulas (harder, more clauses, cap at 5000)
    if dense_count > 0:
        dense_sizes = [500, 1000, 2000, 3000, 5000] * (dense_count // 5 + 1)
        dense_sizes = dense_sizes[:dense_count]
        for i, size in enumerate(dense_sizes):
            size = min(size, MAX_VARS)
            output = cnf_dir / f"dense_{size}_{i}.cnf"
            generate_large_3sat(size, clause_to_var_ratio=6.0, output_file=str(output))
            total_generated += 1
    
    # 4. Easy-SAT formulas (GPU-friendly, cap at 5000)
    if easy_count > 0:
        easy_sizes = [500, 1000, 2000, 3000, 5000] * (easy_count // 5 + 1)
        easy_sizes = easy_sizes[:easy_count]
        for i, size in enumerate(easy_sizes):
            size = min(size, MAX_VARS)
            output = cnf_dir / f"easy_{size}_{i}.cnf"
            generate_easy_sat_cnf(size, str(output))
            total_generated += 1
    
    # 5. Horn formulas (polynomial-time SAT, cap at 5000)
    if horn_count > 0:
        horn_sizes = [1000, 2000, 3000, 4000, 5000] * (horn_count // 5 + 1)
        horn_sizes = horn_sizes[:horn_count]
        for i, size in enumerate(horn_sizes):
            size = min(size, MAX_VARS)
            output = cnf_dir / f"horn_{size}_{i}.cnf"
            generate_horn_cnf(size, str(output))
            total_generated += 1
    
    # 6. k-SAT formulas (various k values, cap at 5000)
    if ksat_count > 0:
        k_values = [4, 5, 6, 7, 8] * (ksat_count // 5 + 1)
        var_sizes = [500, 1000, 2000, 3000, 5000] * (ksat_count // 5 + 1)
        for i in range(ksat_count):
            k = k_values[i % len(k_values)]
            size = min(var_sizes[i % len(var_sizes)], MAX_VARS)
            output = cnf_dir / f"ksat{k}_{size}_{i}.cnf"
            generate_ksat_cnf(size, k, str(output))
            total_generated += 1
    
    # 7. Random CNF formulas (variable clause sizes, cap at 5000)
    if random_count > 0:
        random_sizes = [500, 1000, 2000, 3000, 5000] * (random_count // 5 + 1)
        random_sizes = random_sizes[:random_count]
        for i, size in enumerate(random_sizes):
            size = min(size, MAX_VARS)
            output = cnf_dir / f"random_{size}_{i}.cnf"
            generate_random_cnf(size, str(output))
            total_generated += 1
    
    return total_generated


def main():
    parser = argparse.ArgumentParser(description="Generate CNF files for benchmarking and testing (max 5000 variables)", formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument(
        '--standard-count',
        type=int,
        default=4,
        help='Number of standard 3-SAT instances to generate (default: 4)'
    )
    parser.add_argument(
        '--large-count',
        type=int,
        default=4,
        help='Number of large 3-SAT instances to generate (default: 4)'
    )
    parser.add_argument(
        '--dense-count',
        type=int,
        default=3,
        help='Number of dense formulas to generate (default: 3)'
    )
    parser.add_argument(
        '--easy-count',
        type=int,
        default=4,
        help='Number of easy-SAT formulas to generate (default: 4)'
    )
    parser.add_argument(
        '--horn-count',
        type=int,
        default=4,
        help='Number of Horn formulas to generate (default: 4)'
    )
    parser.add_argument(
        '--ksat-count',
        type=int,
        default=4,
        help='Number of k-SAT formulas to generate (default: 4)'
    )
    parser.add_argument(
        '--random-count',
        type=int,
        default=4,
        help='Number of random CNFs with variable clause sizes to generate (default: 4)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--dir',
        type=str,
        default=None,
        help='Output directory (default: ./cnfs/)'
    )
    
    args = parser.parse_args()
    
    # set random seed
    random.seed(args.seed)
    
    # get output directory
    if args.dir:
        cnf_dir = Path(args.dir)
    else:
        cnf_dir = Path(__file__).parent / "cnfs"
    
    # generate CNFs
    total = generate_all(
        cnf_dir,
        standard_count=args.standard_count,
        large_count=args.large_count,
        dense_count=args.dense_count,
        easy_count=args.easy_count,
        horn_count=args.horn_count,
        ksat_count=args.ksat_count,
        random_count=args.random_count
    )
    

if __name__ == "__main__":
    main()
