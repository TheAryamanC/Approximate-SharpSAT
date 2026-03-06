"""XGBoost model training pipeline for XOR hash prediction"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import argparse
from pathlib import Path
from typing import Dict, List


class CNFFeatureExtractor:
    """Extract comprehensive features from CNF formulas for XGBoost training"""
    
    @staticmethod
    def parse_cnf_file(cnf_path: str) -> Dict:
        """Parse a CNF file in DIMACS format"""
        clauses = []
        num_variables = 0
        num_clauses = 0
        
        with open(cnf_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('c'):
                    continue
                if line.startswith('p'):
                    parts = line.split()
                    num_variables = int(parts[2])
                    num_clauses = int(parts[3])
                    continue
                
                literals = list(map(int, line.split()))
                if literals[-1] == 0:
                    literals = literals[:-1]
                if literals:
                    clauses.append(literals)
        
        return {
            'num_variables': num_variables,
            'num_clauses': len(clauses),
            'clauses': clauses
        }
    
    @staticmethod
    def extract_features(cnf: Dict) -> np.ndarray:
        """Extract 22 features the CNF to learn optimal sparsity"""
        num_vars = cnf['num_variables']
        num_clauses = cnf['num_clauses']
        clauses = cnf['clauses']
        
        features = []
        
        features.append(num_vars)                           # 1. number of variables
        features.append(num_clauses)                        # 2. number of clauses
        features.append(num_clauses / max(1, num_vars))     # 3. clause-to-variable ratio
        features.append(num_vars / max(1, num_clauses))     # 4. variable-to-clause ratio
        features.append(np.log1p(num_vars))                 # 5. log(number of variables)
        features.append(np.log1p(num_clauses))              # 6. log(number of clauses)
        
        clause_sizes = [len(c) for c in clauses]
        if clause_sizes:
            features.extend([
                np.mean(clause_sizes),                      # 7. average clause size
                np.std(clause_sizes),                       # 8. std deviation of clause size
                np.max(clause_sizes),                       # 9. max clause size
                np.min(clause_sizes),                       # 10. min clause size
                np.median(clause_sizes)                     # 11. median clause size
            ])
        else:
            features.extend([0, 0, 0, 0, 0]) # fallback for empty CNF
        
        var_occurrences = [0] * (num_vars + 1)
        positive_occurrences = [0] * (num_vars + 1)
        negative_occurrences = [0] * (num_vars + 1)
        
        for clause in clauses:
            for lit in clause:
                var = abs(lit)
                if var <= num_vars:
                    var_occurrences[var] += 1
                    if lit > 0:
                        positive_occurrences[var] += 1
                    else:
                        negative_occurrences[var] += 1
        
        var_occ_list = var_occurrences[1:]
        if var_occ_list:
            mean_occ = np.mean(var_occ_list)
            features.extend([
                mean_occ,                                   # 12. average variable occurrence
                np.std(var_occ_list),                       # 13. std deviation of variable occurrence
                np.max(var_occ_list),                       # 14. max variable occurrence
                np.min(var_occ_list),                       # 15. min variable occurrence
                np.median(var_occ_list),                    # 16. median variable occurrence
                np.std(var_occ_list) / max(mean_occ, 1e-6)  # 17. coefficient of variation of variable occurrence
            ])
        else:
            features.extend([0, 0, 0, 0, 0, 0]) # fallback for no variables
        
        polarity_balances = []
        for var in range(1, num_vars + 1):
            total = var_occurrences[var]
            if total > 0:
                balance = positive_occurrences[var] / total
                polarity_balances.append(balance)
        
        if polarity_balances:
            biased = sum(1 for b in polarity_balances if b > 0.8 or b < 0.2)
            features.extend([
                np.mean(polarity_balances),                 # 18. average polarity balance
                np.std(polarity_balances),                  # 19. std deviation of polarity balance
                biased / max(len(polarity_balances), 1)     # 20. fraction of biased variables    
            ])
        else:
            features.extend([0.5, 0, 0])
        
        horn_clauses = sum(1 for clause in clauses if sum(1 for lit in clause if lit > 0) <= 1)
        features.append(horn_clauses / max(num_clauses, 1)) # 21. horn clause ratio
        
        total_literals = sum(len(c) for c in clauses)
        features.append(total_literals / max(num_vars * num_clauses, 1)) # 22. literal density
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get names of all features for interpretability"""
        return [
            'num_variables', 'num_clauses', 'clause_var_ratio', 'var_clause_ratio',
            'log_num_vars', 'log_num_clauses',
            'avg_clause_size', 'std_clause_size', 'max_clause_size', 
            'min_clause_size', 'median_clause_size',
            'avg_var_occurrence', 'std_var_occurrence', 'max_var_occurrence', 
            'min_var_occurrence', 'median_var_occurrence', 'cv_var_occurrence',
            'avg_polarity_balance', 'std_polarity_balance', 'frac_biased_vars',
            'horn_clause_ratio', 'literal_density'
        ]


def compute_optimal_sparsity_heuristic(features_dict: dict) -> float:
    """Compute optimal sparsity using CNF features with a heuristic formula"""

    # 1. Extract features
    n = features_dict.get('num_variables', 1)
    m_n_ratio = features_dict.get('clause_var_ratio', 4.26)
    avg_k = features_dict.get('avg_clause_size', 3.0)
    horn_p = features_dict.get('horn_clause_ratio', 0.0)

    # 2. Base Sparsity: sqrt(n) scaling with clamping
    base_sparsity = np.clip(3.0 / np.sqrt(n), 0.2, 0.7)

    # 3. Continuous Density Factor - sigmoid adjustment based on clause-to-variable ratio (higher ratio = more constrained = lower sparsity)
    density_factor = 1.1 - (0.35 / (1 + np.exp(-2 * (m_n_ratio - 4.0))))

    # 4. Continuous Size Factor
    size_factor = 1.0 - 0.1 * np.log10(n / 1000.0)
    size_factor = np.clip(size_factor, 0.85, 1.25)

    # 5. Continuous Clause Size Factor - larger average clause size may allow for slightly higher sparsity
    clause_size_factor = 1.0 + 0.05 * (avg_k - 3.0)
    clause_size_factor = np.clip(clause_size_factor, 0.9, 1.1)

    # 6. Continuous Horn Adjustment - more Horn clauses = simpler structure = lower sparsity
    horn_factor = 1.0 - (0.15 * (horn_p ** 2))

    # 7. Combine and Add Noise
    optimal_sparsity = (base_sparsity * density_factor * size_factor * clause_size_factor * horn_factor)

    # Add 2% Gaussian noise for exploration variety
    noise = np.random.normal(0, 0.02)
    
    # Final Clamp to valid probability range
    return float(np.clip(optimal_sparsity + noise, 0.1, 0.9))


def load_benchmark_data(benchmark_dir: str) -> pd.DataFrame:
    """Load and process real benchmark CNF files"""
    cnf_files = sorted(Path(benchmark_dir).glob('*.cnf'))
    
    data = []
    feature_names = CNFFeatureExtractor.get_feature_names()
    
    for i, cnf_path in enumerate(cnf_files):
        try:
            cnf_dict = CNFFeatureExtractor.parse_cnf_file(str(cnf_path))
            features = CNFFeatureExtractor.extract_features(cnf_dict)
            features_dict = dict(zip(feature_names, features))
            
            optimal_sparsity = compute_optimal_sparsity_heuristic(features_dict)
            
            sample = {'optimal_sparsity': optimal_sparsity}
            for fname, fval in zip(feature_names, features):
                sample[fname] = fval
            
            data.append(sample)
            
        except Exception as e:
            continue
    
    return pd.DataFrame(data)


def train_model(X: np.ndarray, y: np.ndarray, feature_names: list) -> xgb.XGBRegressor:
    """Train XGBoost regression model"""
    
    # create train/test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 300,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.05,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # train
    model = xgb.XGBRegressor(**params)
    
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    
    # evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_metrics = {
        'mse': mean_squared_error(y_train, y_train_pred),
        'mae': mean_absolute_error(y_train, y_train_pred),
        'r2': r2_score(y_train, y_train_pred)
    }
    
    test_metrics = {
        'mse': mean_squared_error(y_test, y_test_pred),
        'mae': mean_absolute_error(y_test, y_test_pred),
        'r2': r2_score(y_test, y_test_pred)
    }
    
    # feature importance
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost model for XOR hash prediction on real benchmark data')
    parser.add_argument('--benchmark-dir', default='./training_cnfs', help='Directory containing benchmark CNF files')
    parser.add_argument('--model-output', default='model.pkl', help='Output path for trained model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    script_dir = Path(__file__).parent
    benchmark_dir = (script_dir / args.benchmark_dir).resolve()
    model_output = script_dir / args.model_output
    
    # load benchmark data
    if not benchmark_dir.exists():
        return 1
    
    df_training = load_benchmark_data(str(benchmark_dir))
    
    if len(df_training) == 0:
        return 1
    
    # prepare features and targets
    feature_cols = [c for c in df_training.columns if c != 'optimal_sparsity']
    X = df_training[feature_cols].values
    y = df_training['optimal_sparsity'].values
    
    # train model
    model = train_model(X, y, feature_cols)
    
    # save model
    joblib.dump(model, model_output)
    
    return 0


if __name__ == '__main__':
    exit(main())
