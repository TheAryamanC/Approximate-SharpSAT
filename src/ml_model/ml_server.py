"""Persistent ML predictor server that loads the XGBoost model once and serves predictions on demand"""

import sys
import os
import numpy as np
import joblib
from pathlib import Path

def load_model():
    """Load the XGBoost model once at startup"""
    script_dir = Path(__file__).parent
    model_path = script_dir / 'model.pkl'
    
    if not model_path.exists():
        print("ERROR: Model file not found", file=sys.stderr, flush=True)
        sys.exit(1)
    
    try:
        model = joblib.load(str(model_path))
        print("READY", flush=True)  # signal to C++ that we're ready
        return model
    except Exception as e:
        sys.exit(1)

def construct_feature_vector(features):
    """Construct 22-dimensional feature vector"""
    num_vars = features[0]
    num_clauses = features[1]
    avg_clause_size = features[2]
    var_clause_ratio = features[3]
    var_occurrences = features[4:] if len(features) > 4 else []
    
    feature_vec = []
    
    # basic features
    feature_vec.append(num_vars)
    feature_vec.append(num_clauses)
    feature_vec.append(num_clauses / max(1, num_vars))
    feature_vec.append(var_clause_ratio)
    feature_vec.append(np.log1p(num_vars))
    feature_vec.append(np.log1p(num_clauses))
    
    # clause size statistics
    feature_vec.extend([avg_clause_size, 0.5, avg_clause_size, avg_clause_size, avg_clause_size])
    
    # variable occurrences
    if var_occurrences and len(var_occurrences) > 0:
        var_occurrences = np.array(var_occurrences)
        mean_occ = np.mean(var_occurrences)
        feature_vec.append(mean_occ)
        feature_vec.append(np.std(var_occurrences))
        feature_vec.append(np.max(var_occurrences))
        feature_vec.append(np.min(var_occurrences))
        feature_vec.append(np.median(var_occurrences))
        feature_vec.append(np.std(var_occurrences) / max(mean_occ, 1e-6))
    else:
        avg_occ = num_clauses * avg_clause_size / max(num_vars, 1)
        feature_vec.extend([avg_occ, avg_occ * 0.3, avg_occ * 1.5, avg_occ * 0.5, avg_occ, 0.3])
    
    # other features
    feature_vec.extend([0.5, 0.1, 0.2])  # polarity balance
    feature_vec.append(0.1)              # horn clause ratio
    literal_density = avg_clause_size / max(num_vars, 1)
    feature_vec.append(literal_density)
    
    return np.array(feature_vec, dtype=np.float32)

# Convert predicted sparsity to variable importance scores
def predict_importance_scores(model, num_vars, features):
    """Predict variable importance scores from 0.0 to 1.0"""
    feature_vec = construct_feature_vector(features)
    
    try:
        # predict sparsity
        sparsity = model.predict(feature_vec.reshape(1, -1))[0]
        sparsity = max(0.15, min(0.85, float(sparsity)))
        
        # convert to importance scores based on occurrences
        var_occurrences = features[4:] if len(features) > 4 else []
        
        if var_occurrences and len(var_occurrences) >= num_vars:
            # mormalize occurrences to importance scores
            occurrences = np.array(var_occurrences[:num_vars])
            max_occ = np.max(occurrences) if occurrences.size > 0 else 1
            
            if max_occ > 0:
                # Scale to [0.2, 0.8] based on occurrence frequency
                importance = 0.2 + 0.6 * (occurrences / max_occ)
            else:
                importance = np.full(num_vars, 0.5)
        else:
            # fallback is uniform importance
            importance = np.full(num_vars, 0.5)
        
        return importance
    except Exception as e:
        return np.full(num_vars, 0.5)

def main():
    """Main server loop: load model once, serve predictions"""
    model = load_model()
    
    # process prediction requests from stdin
    for line in sys.stdin:
        line = line.strip()
        if not line or line == "QUIT":
            break
        
        try:
            # parse: num_vars num_clauses avg_clause_size var_clause_ratio [var_occurrences...]
            features = list(map(float, line.split()))
            
            if len(features) < 4:
                print("ERROR: Not enough features", file=sys.stderr, flush=True)
                continue
            
            num_vars = int(features[0])
            
            # predict importance scores
            importance = predict_importance_scores(model, num_vars, features)
            
            # output importance scores (one per line, then END marker)
            for score in importance:
                print(score, flush=True)
            print("END", flush=True)
            
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr, flush=True)
            print("ERROR", flush=True)

if __name__ == "__main__":
    main()
