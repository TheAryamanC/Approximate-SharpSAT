"""
This script is called by the C++ program to predict optimal XOR sparsity based on features extracted from the CNF formula, using a pre-trained XGBoost model to make the prediction
"""

import numpy as np
import sys
import os
from pathlib import Path
import joblib


# Parse features from the input file
def parse_features_file(features_file: str) -> dict:
    """Parse features from the input file (2 lines: basic features + variable occurrences)"""
    with open(features_file, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 1:
        raise ValueError("Features file is empty")
    
    # parse basic features
    basic_features = list(map(float, lines[0].strip().split()))
    
    if len(basic_features) < 4:
        raise ValueError("Not enough basic features")
    
    num_variables = int(basic_features[0])
    num_clauses = int(basic_features[1])
    avg_clause_size = basic_features[2]
    var_clause_ratio = basic_features[3]
    
    # parse variable occurrences if provided
    var_occurrences = []
    if len(lines) > 1:
        var_occurrences = list(map(int, lines[1].strip().split()))
    
    return {
        'num_variables': num_variables,
        'num_clauses': num_clauses,
        'avg_clause_size': avg_clause_size,
        'var_clause_ratio': var_clause_ratio,
        'var_occurrences': var_occurrences
    }


# Predict sparsity using the model
def predict_sparsity_ml(features: dict, model_path: str) -> float:
    """Predict sparsity using trained XGBoost model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # load model
    model = joblib.load(model_path)
    
    # construct feature vector from available data
    feature_vector = construct_feature_vector(features)
    
    # predict
    prediction = model.predict(feature_vector.reshape(1, -1))[0]
    
    # clamp if prediction is extreme
    return max(0.15, min(0.85, float(prediction)))


# Construct feature vector from parsed features
def construct_feature_vector(features: dict) -> np.ndarray:
    """Construct feature vector from parsed features"""
    num_vars = features['num_variables']
    num_clauses = features['num_clauses']
    avg_clause_size = features['avg_clause_size']
    var_clause_ratio = features['var_clause_ratio']
    var_occurrences = features['var_occurrences']
    
    feature_vec = []
    
    # basic size features
    feature_vec.append(num_vars)
    feature_vec.append(num_clauses)
    feature_vec.append(num_clauses / max(1, num_vars))  # clause_var_ratio
    feature_vec.append(var_clause_ratio)                # var_clause_ratio
    feature_vec.append(np.log1p(num_vars))
    feature_vec.append(np.log1p(num_clauses))
    
    # clause size statistics
    feature_vec.extend([avg_clause_size, 0.5, avg_clause_size, avg_clause_size, avg_clause_size])
    
    # variable occurrence statistics
    if var_occurrences and len(var_occurrences) > 0:
        feature_vec.append(np.mean(var_occurrences))
        feature_vec.append(np.std(var_occurrences))
        feature_vec.append(np.max(var_occurrences))
        feature_vec.append(np.min(var_occurrences))
        feature_vec.append(np.median(var_occurrences))
        mean_occ = np.mean(var_occurrences)
        feature_vec.append(np.std(var_occurrences) / max(mean_occ, 1e-6))
    else:
        # default values
        avg_occ = num_clauses * avg_clause_size / max(num_vars, 1)
        feature_vec.extend([avg_occ, avg_occ * 0.3, avg_occ * 1.5, avg_occ * 0.5, avg_occ, 0.3])
    
    # estimate rest of features with defaults (since we don't have full CNF data here)
    feature_vec.extend([0.5, 0.1, 0.2])                  # polarity balance
    feature_vec.append(0.1)                              # horn clause ratio
    literal_density = avg_clause_size / max(num_vars, 1) # literal density
    feature_vec.append(literal_density)
    
    return np.array(feature_vec, dtype=np.float32)


def predict_hash_sparsity(features_file: str) -> float:
    """Main prediction function called by C++ code"""
    # parse features
    features = parse_features_file(features_file)
    
    # find model
    script_dir = Path(__file__).parent
    model_path = script_dir / 'model.pkl'
    
    # return prediction
    return predict_sparsity_ml(features, str(model_path))   


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hash_predictor.py <features_file>", file=sys.stderr)
        sys.exit(1)
    
    features_file = sys.argv[1]
    predict_hash_sparsity(features_file)
