#!/usr/bin/env python3
"""
Comprehensive benchmarking script for SharpSAT
Runs multiple trials to measure:
1. Variance with/without ML
2. GPU vs CPU performance
3. Identifies scenarios where GPU wins
"""

import subprocess
import json
import statistics
import time
import os
import sys
from pathlib import Path

class BenchmarkRunner:
    def __init__(self, executable_path, cnf_dir):
        self.executable = executable_path
        self.cnf_dir = Path(cnf_dir)
        self.results = []
        
    def run_single_trial(self, cnf_file, use_ml=False, use_cuda=True, epsilon=0.8, delta=0.2, seed=None):
        """Run a single trial and extract timing and count information"""
        cmd = [self.executable, str(cnf_file), 
               "--epsilon", str(epsilon), 
               "--delta", str(delta)]
        
        if use_ml:
            cmd.append("--use-ml")
        if not use_cuda:
            cmd.append("--no-cuda")
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
            
        try:
            start = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)  # 30 second timeout per trial
            wall_time = time.time() - start
            
            if result.returncode != 0:
                print(f"\n    Return code: {result.returncode}")
                if result.stderr:
                    print(f"    Stderr: {result.stderr[:200]}")
                return None
                
            # Parse output
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
                'success': count is not None
            }
        except subprocess.TimeoutExpired:
            print(f"\n    Timeout after 30s")
            return None
        except Exception as e:
            print(f"Error running trial: {e}")
            return None
            
    def run_trials(self, cnf_file, num_trials=10, **kwargs):
        """Run multiple trials and collect statistics"""
        print(f"\nRunning {num_trials} trials for {cnf_file.name}")
        print(f"  Config: ML={kwargs.get('use_ml', False)}, CUDA={kwargs.get('use_cuda', True)}")
        
        results = []
        times = []
        counts = []
        iterations_list = []
        
        for i in range(num_trials):
            print(f"  Trial {i+1}/{num_trials}...", end='', flush=True)
            # Use different seed for each trial to observe variance
            trial_seed = 42 + i
            result = self.run_single_trial(cnf_file, seed=trial_seed, **kwargs)
            if result and result['success']:
                results.append(result)
                times.append(result['time'])
                if result['count'] is not None:
                    counts.append(result['count'])
                    # Show the count for this trial
                    count_str = f"{result['count']:.2e}" if result['count'] > 1e6 else f"{result['count']:.0f}"
                    print(f" ✓ (count: {count_str}, time: {result['time']:.3f}s)")
                else:
                    print(" ✓")
                if result['iterations'] is not None:
                    iterations_list.append(result['iterations'])
            else:
                print(" ✗ (failed)")
                
        if not results:
            return None
            
        # Compute statistics
        stats = {
            'cnf_file': str(cnf_file.name),
            'num_successful_trials': len(results),
            'config': kwargs,
            'time_stats': {
                'mean': statistics.mean(times),
                'median': statistics.median(times),
                'stdev': statistics.stdev(times) if len(times) > 1 else 0,
                'variance': statistics.variance(times) if len(times) > 1 else 0,
                'min': min(times),
                'max': max(times),
                'cv': (statistics.stdev(times) / statistics.mean(times) * 100) if len(times) > 1 and statistics.mean(times) > 0 else 0
            }
        }
        
        if counts:
            # Calculate variance using log-scale for large numbers to avoid overflow
            import math
            mean_count = statistics.mean(counts)
            if mean_count > 0:
                # Use log-scale for very large numbers
                if mean_count > 1e50:
                    # Filter out zeros and compute log only for valid values
                    log_counts = [math.log10(c) for c in counts if c > 0]
                    # Check for inf/nan
                    log_counts = [x for x in log_counts if math.isfinite(x)]
                    
                    if len(log_counts) > 1 and len(set(log_counts)) > 1:
                        log_mean = statistics.mean(log_counts)
                        log_std = statistics.stdev(log_counts)
                        rel_cv = (log_std / log_mean * 100) if log_mean > 0 else 0
                    else:
                        log_std = 0
                        rel_cv = 0
                    
                    stats['count_stats'] = {
                        'mean': mean_count,
                        'median': statistics.median(counts),
                        'log_stdev': log_std,
                        'relative_cv': rel_cv
                    }
                else:
                    if len(counts) > 1 and len(set(counts)) > 1:
                        relative_counts = [c / mean_count for c in counts]
                        rel_std = statistics.stdev(counts)
                        rel_cv = (statistics.stdev(relative_counts) / statistics.mean(relative_counts) * 100)
                    else:
                        rel_std = 0
                        rel_cv = 0
                    
                    stats['count_stats'] = {
                        'mean': mean_count,
                        'median': statistics.median(counts),
                        'stdev': rel_std,
                        'relative_cv': rel_cv
                    }
            else:
                stats['count_stats'] = {'mean': 0}
                
        if iterations_list:
            stats['iterations_stats'] = {
                'mean': statistics.mean(iterations_list),
                'median': statistics.median(iterations_list),
                'stdev': statistics.stdev(iterations_list) if len(iterations_list) > 1 else 0
            }
            
        return stats
        
    def compare_configurations(self, cnf_files, num_trials=10):
        """Compare different configurations on multiple CNF files"""
        all_results = []
        
        configs = [
            {'name': 'ML+CUDA', 'use_ml': True, 'use_cuda': True},
            {'name': 'NoML+CUDA', 'use_ml': False, 'use_cuda': True},
            {'name': 'ML+CPU', 'use_ml': True, 'use_cuda': False},
            {'name': 'NoML+CPU', 'use_ml': False, 'use_cuda': False},
        ]
        
        for cnf_file in cnf_files:
            print(f"\n{'='*70}")
            print(f"Benchmarking: {cnf_file.name}")
            print(f"{'='*70}")
            
            cnf_results = {'cnf_file': str(cnf_file.name), 'configs': {}}
            
            for config in configs:
                config_name = config['name']
                config_params = {k: v for k, v in config.items() if k != 'name'}
                
                # Print which configuration is being tested
                ml_status = "ML enabled" if config_params.get('use_ml', False) else "ML disabled"
                cuda_status = "CUDA enabled" if config_params.get('use_cuda', False) else "CUDA disabled"
                print(f"\nConfiguration: {config_name} ({ml_status}, {cuda_status})")
                
                stats = self.run_trials(cnf_file, num_trials=num_trials, **config_params)
                if stats:
                    cnf_results['configs'][config_name] = stats
                    
            all_results.append(cnf_results)
            
        return all_results
        
    def print_summary(self, results):
        """Print a formatted summary of results"""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        for cnf_result in results:
            cnf_name = cnf_result['cnf_file']
            print(f"\n{cnf_name}:")
            print("-" * 80)
            
            configs = cnf_result['configs']
            if not configs:
                print("  No successful runs")
                continue
                
            # Print comparison table
            print(f"{'Config':<15} {'Time (s)':<15} {'Time CV %':<12} {'Count CV %':<12} {'Speedup':<10}")
            print("-" * 80)
            
            baseline_time = None
            if 'NoML+CPU' in configs:
                baseline_time = configs['NoML+CPU']['time_stats']['mean']
            
            for config_name in ['ML+CUDA', 'NoML+CUDA', 'ML+CPU', 'NoML+CPU']:
                if config_name not in configs:
                    continue
                    
                stats = configs[config_name]
                time_mean = stats['time_stats']['mean']
                time_cv = stats['time_stats']['cv']
                count_cv = stats.get('count_stats', {}).get('relative_cv', 0)
                
                speedup = ""
                if baseline_time and baseline_time > 0:
                    speedup = f"{baseline_time / time_mean:.2f}x"
                    
                print(f"{config_name:<15} {time_mean:<15.3f} {time_cv:<12.2f} {count_cv:<12.2f} {speedup:<10}")
                
        # Print variance comparison
        print("\n" + "="*80)
        print("ML vs Non-ML VARIANCE COMPARISON")
        print("="*80)
        
        ml_variances = []
        no_ml_variances = []
        
        for cnf_result in results:
            configs = cnf_result['configs']
            
            # Compare ML+CUDA vs NoML+CUDA
            if 'ML+CUDA' in configs and 'NoML+CUDA' in configs:
                ml_cv = configs['ML+CUDA'].get('count_stats', {}).get('relative_cv', 0)
                no_ml_cv = configs['NoML+CUDA'].get('count_stats', {}).get('relative_cv', 0)
                
                if ml_cv > 0 or no_ml_cv > 0:
                    ml_variances.append(ml_cv)
                    no_ml_variances.append(no_ml_cv)
                    
                    print(f"{cnf_result['cnf_file']}:")
                    print(f"  ML Count CV:    {ml_cv:.2f}%")
                    print(f"  Non-ML Count CV: {no_ml_cv:.2f}%")
                    improvement = ((no_ml_cv - ml_cv) / no_ml_cv * 100) if no_ml_cv > 0 else 0
                    print(f"  Improvement:     {improvement:.1f}%")
                    print()
                    
        if ml_variances and no_ml_variances:
            print(f"Average ML CV: {statistics.mean(ml_variances):.2f}%")
            print(f"Average Non-ML CV: {statistics.mean(no_ml_variances):.2f}%")
            avg_improvement = statistics.mean([(no_ml_variances[i] - ml_variances[i]) / no_ml_variances[i] * 100 
                                              if no_ml_variances[i] > 0 else 0 
                                              for i in range(len(ml_variances))])
            print(f"Average Improvement: {avg_improvement:.1f}%")
            
        # GPU vs CPU comparison
        print("\n" + "="*80)
        print("GPU vs CPU PERFORMANCE")
        print("="*80)
        
        for cnf_result in results:
            configs = cnf_result['configs']
            
            if 'NoML+CUDA' in configs and 'NoML+CPU' in configs:
                cuda_time = configs['NoML+CUDA']['time_stats']['mean']
                cpu_time = configs['NoML+CPU']['time_stats']['mean']
                speedup = cpu_time / cuda_time if cuda_time > 0 else 0
                
                print(f"{cnf_result['cnf_file']}:")
                print(f"  CPU Time:  {cpu_time:.3f}s")
                print(f"  GPU Time:  {cuda_time:.3f}s")
                if speedup >= 1.0:
                    print(f"  Result:    GPU is {speedup:.2f}x FASTER ✓")
                else:
                    print(f"  Result:    CPU is {1/speedup:.2f}x faster (GPU overhead)")
                print()

def main():
    # Find executable
    build_dir = Path(__file__).parent.parent / "build"
    executable = build_dir / "sharp_sat"
    
    if not executable.exists():
        print(f"Error: Executable not found at {executable}")
        sys.exit(1)
        
    # Find CNF files
    cnf_dir = Path(__file__).parent / "cnfs"
    cnf_files = sorted(cnf_dir.glob("*.cnf"))
    
    if not cnf_files:
        print(f"Error: No CNF files found in {cnf_dir}")
        sys.exit(1)
    
    # Use ALL CNF files in the directory
    print("SharpSAT Comprehensive Benchmark")
    print("="*80)
    print(f"Executable: {executable}")
    print(f"CNF files: {len(cnf_files)}")
    print(f"Files to test: {', '.join([f.name for f in cnf_files])}")
    print(f"Trials per config: 4")
    print("="*80)
    
    runner = BenchmarkRunner(str(executable), cnf_dir)
    results = runner.compare_configurations(cnf_files, num_trials=4)
    
    # Print summary
    runner.print_summary(results)
    
    # Save results to JSON
    output_file = build_dir / "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
