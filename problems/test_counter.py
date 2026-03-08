import subprocess
import json
import statistics
import time
import os
import sys
from pathlib import Path
import math
import random
import numpy as np
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

class CNFGenerator:
    """Generate CNF files for testing"""
    
    @staticmethod
    def generate_3sat(num_vars: int, num_clauses: int, output_file: str, seed: int = 42):
        """Generate a random 3-SAT instance"""
        random.seed(seed)
        
        with open(output_file, 'w') as f:
            f.write(f"c Random 3-SAT formula\n")
            f.write(f"c Generated with seed {seed}\n")
            f.write(f"p cnf {num_vars} {num_clauses}\n")
            
            for i in range(num_clauses):
                vars_in_clause = random.sample(range(1, num_vars + 1), min(3, num_vars))
                clause = [v * random.choice([1, -1]) for v in vars_in_clause]
                f.write(' '.join(map(str, clause)) + ' 0\n')
    
    @staticmethod
    def generate_easy_sat(num_vars: int, output_file: str, seed: int = 42):
        """Generate easy-SAT CNF"""
        random.seed(seed)
        num_clauses = int(num_vars * 2.0)
        clauses = []
        
        # Add unit clauses to make it easier
        for i in range(min(10, num_vars)):
            clauses.append([i + 1])
        
        for _ in range(num_clauses - 10):
            clause_size = random.randint(4, min(6, num_vars))
            vars_in_clause = random.sample(range(1, num_vars + 1), clause_size)
            literals = [v if random.random() > 0.5 else -v for v in vars_in_clause]
            clauses.append(literals)
        
        with open(output_file, 'w') as f:
            f.write(f"c Easy-SAT formula\n")
            f.write(f"p cnf {num_vars} {len(clauses)}\n")
            for clause in clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")
    
    @staticmethod
    def generate_horn(num_vars: int, output_file: str, seed: int = 42):
        """Generate Horn CNF"""
        random.seed(seed)
        num_clauses = int(num_vars * 2.5)
        clauses = []
        
        for _ in range(num_clauses):
            clause_size = random.randint(2, min(4, num_vars))
            vars_selected = random.sample(range(1, num_vars + 1), clause_size)
            has_positive = random.random() > 0.3
            literals = []
            for i, v in enumerate(vars_selected):
                if has_positive and i == 0:
                    literals.append(v)
                else:
                    literals.append(-v)
            clauses.append(literals)
        
        with open(output_file, 'w') as f:
            f.write(f"c Horn formula\n")
            f.write(f"p cnf {num_vars} {len(clauses)}\n")
            for clause in clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")
    
    @staticmethod
    def generate_ksat(num_vars: int, k: int, output_file: str, seed: int = 42):
        """Generate k-SAT CNF"""
        random.seed(seed)
        num_clauses = int(num_vars * 4.0)
        k_actual = min(k, num_vars)
        clauses = []
        
        for _ in range(num_clauses):
            vars_selected = random.sample(range(1, num_vars + 1), k_actual)
            literals = [v if random.random() > 0.5 else -v for v in vars_selected]
            clauses.append(literals)
        
        with open(output_file, 'w') as f:
            f.write(f"c {k}-SAT formula\n")
            f.write(f"p cnf {num_vars} {len(clauses)}\n")
            for clause in clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")
    
    @staticmethod
    def generate_random(num_vars: int, output_file: str, seed: int = 42):
        """Generate random CNF with variable-sized clauses"""
        random.seed(seed)
        num_clauses = int(num_vars * 3.5)
        clauses = []
        
        for _ in range(num_clauses):
            clause_size = random.randint(2, min(7, num_vars))
            vars_selected = random.sample(range(1, num_vars + 1), clause_size)
            literals = [v if random.random() > 0.5 else -v for v in vars_selected]
            clauses.append(literals)
        
        with open(output_file, 'w') as f:
            f.write(f"c Random CNF\n")
            f.write(f"p cnf {num_vars} {len(clauses)}\n")
            for clause in clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")
    
    @staticmethod
    def generate_large_3sat(num_vars: int, output_file: str, seed: int = 42):
        """Generate large 3-SAT at phase transition"""
        random.seed(seed)
        num_clauses = int(num_vars * 4.26)  # Phase transition ratio
        clauses = []
        
        for _ in range(num_clauses):
            vars_in_clause = random.sample(range(1, num_vars + 1), min(3, num_vars))
            literals = [v if random.random() > 0.5 else -v for v in vars_in_clause]
            clauses.append(literals)
        
        with open(output_file, 'w') as f:
            f.write(f"c Large 3-SAT formula\n")
            f.write(f"p cnf {num_vars} {len(clauses)}\n")
            for clause in clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")
    
    @staticmethod
    def generate_standard_3sat(num_vars: int, output_file: str, seed: int = 42):
        """Generate standard 3-SAT"""
        CNFGenerator.generate_3sat(num_vars, int(num_vars * 4.26), output_file, seed)

class BenchmarkRunner:
    def __init__(self, executable_path, cnf_dir):
        self.executable = executable_path
        self.cnf_dir = Path(cnf_dir)
        self.results = []
        
    def calculate_timeout(self, num_vars: int) -> int:
        """Calculate appropriate timeout based on problem size"""
        # Scale timeout with problem size: larger problems need more time
        if num_vars <= 100:
            return 30
        elif num_vars <= 500:
            return 60
        elif num_vars <= 1000:
            return 120
        elif num_vars <= 2000:
            return 300
        elif num_vars <= 3000:
            return 600
        else:
            return 1200  # 20 minutes for largest problems
        
    def run_single_trial(self, cnf_file, use_ml=False, use_cuda=True, epsilon=0.8, delta=0.2, seed=None, timeout=None):
        """Run a single trial and extract timing and count information"""
        cmd = [self.executable, str(cnf_file), "--epsilon", str(epsilon), "--delta", str(delta)]
        
        if use_ml:
            cmd.append("--use-ml")
        if not use_cuda:
            cmd.append("--no-cuda")
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        
        if timeout is None:
            timeout = 30
            
        try:
            start = time.time()
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=timeout)
            wall_time = time.time() - start
            
            if result.returncode != 0:
                return None
                
            # parse output
            output = result.stdout
            count = None
            reported_time = None
            iterations = None
            lower_bound = None
            upper_bound = None
            
            for line in output.split('\n'):
                if 'Count:' in line:
                    try:
                        count = float(line.split('Count:')[1].strip())
                    except:
                        pass
                elif 'Time:' in line and 'seconds' in line:
                    try:
                        reported_time = float(line.split('Time:')[1].split('seconds')[0].strip())
                    except:
                        pass
                elif 'Iterations:' in line:
                    try:
                        iterations = int(line.split('Iterations:')[1].strip())
                    except:
                        pass
                elif 'Bounds:' in line:
                    try:
                        bounds_str = line.split('Bounds:')[1].strip()
                        bounds_str = bounds_str.strip('[]')
                        parts = bounds_str.split(',')
                        lower_bound = float(parts[0].strip())
                        upper_bound = float(parts[1].strip())
                    except:
                        pass
                        
            return {
                'count': count,
                'time': reported_time if reported_time else wall_time,
                'wall_time': wall_time,
                'iterations': iterations,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'success': count is not None,
                'timeout': False
            }
        except subprocess.TimeoutExpired:
            return {'success': False, 'timeout': True}
        except Exception as e:
            return {'success': False, 'timeout': False}
            
    def run_trials(self, cnf_file, num_trials=10, **kwargs):
        """Run multiple trials and collect statistics"""
        num_vars = self.extract_num_vars(cnf_file)
        timeout = self.calculate_timeout(num_vars)
        
        print(f"  Testing {cnf_file.name} ({num_vars} vars, timeout={timeout}s)")
        
        results = []
        times = []
        counts = []
        iterations_list = []
        
        for i in range(num_trials):
            # Use different seed for each trial
            trial_seed = 1000 + i * 13  # Different seeds to ensure variance
            result = self.run_single_trial(cnf_file, seed=trial_seed, timeout=timeout, **kwargs)
            
            # If any trial times out, skip the entire CNF
            if result and result.get('timeout', False):
                print(f"    ✗ Trial {i+1} timed out - skipping entire CNF")
                return None
            
            if result and result['success']:
                results.append(result)
                times.append(result['time'])
                if result['count'] is not None:
                    counts.append(result['count'])
                if result['iterations'] is not None:
                    iterations_list.append(result['iterations'])
        
        if len(results) < 3:
            return None
        
        # Compute statistics using all data
        stats = {
            'cnf_file': str(cnf_file.name),
            'num_vars': num_vars,
            'num_successful_trials': len(results),
            'config': kwargs,
            'time_stats': {
                'mean': statistics.mean(times),
                'median': statistics.median(times),
                'stdev': statistics.stdev(times) if len(times) > 1 else 0,
                'min': min(times),
                'max': max(times),
            }
        }
        
        if counts:
            # For very large counts, use log-scale to avoid overflow
            mean_count = statistics.mean(counts)
            try:
                if mean_count > 1e100 or any(c > 1e100 for c in counts):
                    # Use log-scale for extremely large numbers
                    log_counts = [math.log10(c) if c > 0 else 0 for c in counts]
                    stats['count_stats'] = {
                        'mean': mean_count,
                        'median': statistics.median(counts),
                        'log_stdev': statistics.stdev(log_counts) if len(log_counts) > 1 else 0,
                        'min': min(counts),
                        'max': max(counts),
                    }
                else:
                    stats['count_stats'] = {
                        'mean': mean_count,
                        'median': statistics.median(counts),
                        'stdev': statistics.stdev(counts) if len(counts) > 1 else 0,
                        'min': min(counts),
                        'max': max(counts),
                    }
            except (OverflowError, ValueError):
                # If we still get overflow, just report mean/median without stdev
                stats['count_stats'] = {
                    'mean': mean_count,
                    'median': statistics.median(counts),
                    'min': min(counts),
                    'max': max(counts),
                }
        
        if iterations_list:
            stats['iterations_stats'] = {
                'mean': statistics.mean(iterations_list),
                'median': statistics.median(iterations_list),
            }
        
        print(f"    ✓ {len(times)}/{num_trials} trials succeeded (avg time: {stats['time_stats']['mean']:.2f}s)")
            
        return stats
    
    def extract_num_vars(self, cnf_file: Path) -> int:
        """Extract number of variables from CNF filename or file"""
        # Try to parse from filename first (e.g., "easy_1000_0.cnf")
        parts = cnf_file.stem.split('_')
        for part in parts:
            if part.isdigit():
                return int(part)
        
        # Fall back to reading the file
        try:
            with open(cnf_file, 'r') as f:
                for line in f:
                    if line.startswith('p cnf'):
                        return int(line.split()[2])
        except:
            pass
        
        return 100  # default fallback
        
    def compare_configurations(self, cnf_files, num_trials=10):
        """Compare different configurations on multiple CNF files"""
        all_results = []
        
        configs = [
            {'name': 'ML+CUDA', 'use_ml': True, 'use_cuda': True},
            {'name': 'NoML+CUDA', 'use_ml': False, 'use_cuda': True},
            {'name': 'ML+CPU', 'use_ml': True, 'use_cuda': False},
            {'name': 'NoML+CPU', 'use_ml': False, 'use_cuda': False},
        ]
        
        total_tests = len(cnf_files) * len(configs)
        current_test = 0
        
        for cnf_file in cnf_files:
            cnf_results = {'cnf_file': str(cnf_file.name), 'configs': {}}
            
            for config in configs:
                current_test += 1
                config_name = config['name']
                config_params = {k: v for k, v in config.items() if k != 'name'}
                
                print(f"\n[{current_test}/{total_tests}] Testing {config_name} on {cnf_file.name}")
                
                stats = self.run_trials(cnf_file, num_trials=num_trials, **config_params)
                if stats:
                    cnf_results['configs'][config_name] = stats
                else:
                    print(f"    ✗ Failed to get results for {config_name}")
                    
            all_results.append(cnf_results)
            
        return all_results
    
    def generate_graphs(self, results: List[Dict], output_dir: Path):
        """Generate comprehensive graphs from results"""
        output_dir.mkdir(exist_ok=True)
        
        print("\nGenerating graphs...")
        
        # Organize results by CNF type and size
        organized = self.organize_results(results)
        
        # 1. Time vs Size for each CNF type and configuration
        self.plot_time_vs_size(organized, output_dir)
        
        # 2. Configuration comparison for each CNF type
        self.plot_config_comparison(organized, output_dir)
        
        # 3. Speedup analysis
        self.plot_speedup_analysis(organized, output_dir)
        
        print(f"Graphs saved to {output_dir}")
    
    def organize_results(self, results: List[Dict]) -> Dict:
        """Organize results by CNF type"""
        organized = {}
        
        for result in results:
            cnf_name = result['cnf_file']
            
            # Extract CNF type from filename (e.g., "easy_1000_0.cnf" -> "easy")
            cnf_type = cnf_name.split('_')[0]
            
            if cnf_type not in organized:
                organized[cnf_type] = []
            
            organized[cnf_type].append(result)
        
        # Sort by size within each type
        for cnf_type in organized:
            organized[cnf_type].sort(key=lambda x: x['configs'].get('ML+CUDA', {}).get('num_vars', 0))
        
        return organized
    
    def plot_time_vs_size(self, organized: Dict, output_dir: Path):
        """Plot execution time vs problem size for each configuration"""
        configs = ['ML+CUDA', 'NoML+CUDA', 'ML+CPU', 'NoML+CPU']
        colors = ['blue', 'green', 'red', 'orange']
        markers = ['o', 's', '^', 'd']
        
        for cnf_type in organized:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for config, color, marker in zip(configs, colors, markers):
                sizes = []
                times = []
                errors = []
                
                for result in organized[cnf_type]:
                    if config in result['configs']:
                        stats = result['configs'][config]
                        sizes.append(stats['num_vars'])
                        times.append(stats['time_stats']['mean'])
                        errors.append(stats['time_stats']['stdev'])
                
                if sizes:
                    ax.errorbar(sizes, times, yerr=errors, label=config, 
                               color=color, marker=marker, markersize=8, 
                               linewidth=2, capsize=5, alpha=0.7)
            
            ax.set_xlabel('Number of Variables', fontsize=14, fontweight='bold')
            ax.set_ylabel('Execution Time (seconds)', fontsize=14, fontweight='bold')
            ax.set_title(f'{cnf_type.upper()} CNF: Execution Time vs Problem Size', 
                        fontsize=16, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'time_vs_size_{cnf_type}.png', dpi=300)
            plt.close()
    
    def plot_config_comparison(self, organized: Dict, output_dir: Path):
        """Plot configuration comparison as bar charts"""
        configs = ['ML+CUDA', 'NoML+CUDA', 'ML+CPU', 'NoML+CPU']
        colors = ['blue', 'green', 'red', 'orange']
        
        for cnf_type in organized:
            # Select a few representative sizes
            sample_results = organized[cnf_type][::len(organized[cnf_type])//5 + 1][:5]
            
            if not sample_results:
                continue
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            x = np.arange(len(sample_results))
            width = 0.2
            
            for i, config in enumerate(configs):
                times = []
                labels = []
                
                for result in sample_results:
                    if config in result['configs']:
                        stats = result['configs'][config]
                        times.append(stats['time_stats']['mean'])
                        labels.append(f"{stats['num_vars']}")
                    else:
                        times.append(0)
                
                ax.bar(x + i * width, times, width, label=config, color=colors[i], alpha=0.8)
            
            ax.set_xlabel('Problem Size (variables)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Execution Time (seconds)', fontsize=14, fontweight='bold')
            ax.set_title(f'{cnf_type.upper()} CNF: Configuration Comparison', 
                        fontsize=16, fontweight='bold')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(labels)
            ax.legend(fontsize=12)
            ax.grid(True, axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'config_comparison_{cnf_type}.png', dpi=300)
            plt.close()
    
    def plot_speedup_analysis(self, organized: Dict, output_dir: Path):
        """Plot speedup analysis (GPU vs CPU, ML vs NoML)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # GPU Speedup (CUDA vs CPU)
        for cnf_type in organized:
            sizes = []
            speedups = []
            
            for result in organized[cnf_type]:
                if 'NoML+CUDA' in result['configs'] and 'NoML+CPU' in result['configs']:
                    cuda_time = result['configs']['NoML+CUDA']['time_stats']['mean']
                    cpu_time = result['configs']['NoML+CPU']['time_stats']['mean']
                    if cuda_time > 0:
                        sizes.append(result['configs']['NoML+CUDA']['num_vars'])
                        speedups.append(cpu_time / cuda_time)
            
            if sizes:
                ax1.plot(sizes, speedups, marker='o', label=cnf_type.upper(), 
                        linewidth=2, markersize=8, alpha=0.7)
        
        ax1.axhline(y=1, color='black', linestyle='--', linewidth=2, label='No speedup')
        ax1.set_xlabel('Number of Variables', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Speedup (CPU/GPU)', fontsize=14, fontweight='bold')
        ax1.set_title('GPU Speedup Analysis', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # ML Impact (NoML vs ML on CUDA)
        for cnf_type in organized:
            sizes = []
            time_ratios = []
            
            for result in organized[cnf_type]:
                if 'ML+CUDA' in result['configs'] and 'NoML+CUDA' in result['configs']:
                    ml_time = result['configs']['ML+CUDA']['time_stats']['mean']
                    noml_time = result['configs']['NoML+CUDA']['time_stats']['mean']
                    if ml_time > 0:
                        sizes.append(result['configs']['ML+CUDA']['num_vars'])
                        time_ratios.append(noml_time / ml_time)
            
            if sizes:
                ax2.plot(sizes, time_ratios, marker='s', label=cnf_type.upper(), 
                        linewidth=2, markersize=8, alpha=0.7)
        
        ax2.axhline(y=1, color='black', linestyle='--', linewidth=2, label='No impact')
        ax2.set_xlabel('Number of Variables', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Time Ratio (NoML/ML)', fontsize=14, fontweight='bold')
        ax2.set_title('ML Impact on Performance', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'speedup_analysis.png', dpi=300)
        plt.close()
        
    def print_summary(self, results):
        """Print a formatted summary of results"""
        print("\n" + "="*100)
        print("COMPREHENSIVE BENCHMARK SUMMARY")
        print("="*100)
        
        # Organize by CNF type
        organized = self.organize_results(results)
        
        for cnf_type in sorted(organized.keys()):
            print(f"\n{'='*100}")
            print(f"{cnf_type.upper()} CNF FORMULAS")
            print(f"{'='*100}")
            
            for result in organized[cnf_type]:
                cnf_name = result['cnf_file']
                configs = result['configs']
                
                if not configs:
                    continue
                
                # Get problem size
                num_vars = configs[list(configs.keys())[0]]['num_vars']
                
                print(f"\n{cnf_name} ({num_vars} variables):")
                print("-" * 100)
                print(f"{'Config':<15} {'Time (s)':<15} {'Std Dev':<15} {'Count':<20} {'Trials':<10}")
                print("-" * 100)
                
                for config_name in ['ML+CUDA', 'NoML+CUDA', 'ML+CPU', 'NoML+CPU']:
                    if config_name not in configs:
                        continue
                    
                    stats = configs[config_name]
                    time_mean = stats['time_stats']['mean']
                    time_std = stats['time_stats']['stdev']
                    count = stats.get('count_stats', {}).get('mean', 0)
                    trials = stats['num_successful_trials']
                    
                    count_str = f"{count:.2e}" if count > 1e6 else f"{count:.0f}"
                    
                    print(f"{config_name:<15} {time_mean:<15.3f} {time_std:<15.3f} {count_str:<20} {trials:<10}")
        
        print("\n" + "="*100)

def main():
    print("="*100)
    print("COMPREHENSIVE MODEL COUNTER BENCHMARK")
    print("="*100)
    
    # Find executable
    bin_dir = Path(__file__).resolve().parent.parent / "bin"
    executable = bin_dir / "approx_counter"
    
    if not executable.exists():
        print(f"Error: Executable not found at {executable}")
        print("Please build the project first using 'make'")
        sys.exit(1)
    
    # Setup CNF directory
    cnf_dir = Path(__file__).resolve().parent / "cnfs_benchmark"
    cnf_dir.mkdir(exist_ok=True)
    
    print(f"Executable: {executable}")
    print(f"CNF Directory: {cnf_dir}")
    print(f"Trials per test: 10")
    print(f"Configurations: 4 (ML+CUDA, NoML+CUDA, ML+CPU, NoML+CPU)")
    print("="*100)
    
    # Generate CNF files
    print("\nGenerating CNF files...")
    cnf_types = ['easy', 'horn', 'random']
    ksat_k_values = [4, 5, 6, 7]
    
    all_cnf_files = []
    generation_seed = 42
    
    # Generate CNFs from 300 to 7500 in increments of 300
    sizes = list(range(300, 7501, 300))
    
    print(f"Generating {len(sizes)} sizes x {len(cnf_types) + len(ksat_k_values)} types = {len(sizes) * (len(cnf_types) + len(ksat_k_values))} CNF files...")
    
    for size in sizes:
        # Easy SAT
        cnf_file = cnf_dir / f"easy_{size}.cnf"
        CNFGenerator.generate_easy_sat(size, str(cnf_file), generation_seed)
        all_cnf_files.append(cnf_file)
        
        # Horn
        cnf_file = cnf_dir / f"horn_{size}.cnf"
        CNFGenerator.generate_horn(size, str(cnf_file), generation_seed)
        all_cnf_files.append(cnf_file)
        
        # Random
        cnf_file = cnf_dir / f"random_{size}.cnf"
        CNFGenerator.generate_random(size, str(cnf_file), generation_seed)
        all_cnf_files.append(cnf_file)
        
        # k-SAT variants
        for k in ksat_k_values:
            cnf_file = cnf_dir / f"ksat{k}_{size}.cnf"
            CNFGenerator.generate_ksat(size, k, str(cnf_file), generation_seed)
            all_cnf_files.append(cnf_file)
    
    print(f"✓ Generated {len(all_cnf_files)} CNF files")
    print(f"  Size range: {min(sizes)} to {max(sizes)} variables")
    print(f"  Types: {', '.join(cnf_types + [f'ksat{k}' for k in ksat_k_values])}")
    
    # Run benchmark
    print("\n" + "="*100)
    print("STARTING BENCHMARK")
    print("="*100)
    print(f"Total tests: {len(all_cnf_files)} files x 4 configs x 10 trials = {len(all_cnf_files) * 4 * 10} runs")
    print("This will take a significant amount of time (hours to days)...")
    print("="*100)
    
    runner = BenchmarkRunner(str(executable), cnf_dir)
    
    start_time = time.time()
    results = runner.compare_configurations(all_cnf_files, num_trials=10)
    total_time = time.time() - start_time
    
    print(f"\n\nBenchmark completed in {total_time/3600:.2f} hours")
    
    # Save results
    output_dir = bin_dir / "benchmark_results"
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {results_file}")
    
    # Generate summary
    runner.print_summary(results)
    
    # Generate graphs
    graphs_dir = output_dir / "graphs"
    runner.generate_graphs(results, graphs_dir)
    
    print("\n" + "="*100)
    print("BENCHMARK COMPLETE!")
    print("="*100)
    print(f"Results directory: {output_dir}")
    print(f"Graphs directory: {graphs_dir}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print("="*100)

if __name__ == "__main__":
    main()
