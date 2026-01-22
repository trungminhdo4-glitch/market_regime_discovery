import json
import numpy as np
import pandas as pd

def compute_regime_bias(labels, forward_returns, config=None):
    """Compute regime statistics with volatility context"""
    if config is None:
        config = {}
    
    unique_labels = np.unique(labels)
    regime_stats = {}
    
    for label in unique_labels:
        mask = labels == label
        if np.sum(mask) == 0:
            continue
            
        regime_returns = forward_returns[mask]
        n_samples = len(regime_returns)
        
        min_size = config.get("min_cluster_size", 100)
        if n_samples < min_size:
            continue
            
        p_up = np.mean(regime_returns > 0)
        sharpe = np.mean(regime_returns) / (np.std(regime_returns) + 1e-8) if np.std(regime_returns) > 0 else 0.0
        
        # 🔥 VOLATILITY CONTEXT
        regime_vol = np.std(regime_returns)
        market_vol = np.std(forward_returns)
        
        regime_stats[str(label)] = {
            "p_up": float(p_up),
            "sharpe": float(sharpe),
            "n_samples": int(n_samples),
            "volatility_regime": float(regime_vol),
            "volatility_market": float(market_vol)
        }
    
    return regime_stats

def compute_regime_durations(labels, timestamps, window_size: int = 48):
    """
    Calculate duration statistics per regime
    timestamps: pandas DatetimeIndex of window start times
    window_size: number of candles per window (for duration calculation)
    """
    if len(labels) != len(timestamps):
        raise ValueError("Labels and timestamps must have same length")
    
    durations = {}
    current_regime = str(labels[0])
    start_time = timestamps[0]
    
    # Track regime segments
    for i in range(1, len(labels)):
        if str(labels[i]) != current_regime:
            # Calculate duration in hours
            end_time = timestamps[i-1]
            duration_hours = (end_time - start_time).total_seconds() / 3600
            duration_hours += window_size / 60  # Add window size in hours (48 candles = 48h for 1h data)
            
            if current_regime not in durations:
                durations[current_regime] = []
            durations[current_regime].append(duration_hours)
            
            # Start new regime
            current_regime = str(labels[i])
            start_time = timestamps[i]
    
    # Handle last segment
    end_time = timestamps[-1]
    duration_hours = (end_time - start_time).total_seconds() / 3600
    duration_hours += window_size / 60
    
    if current_regime not in durations:
        durations[current_regime] = []
    durations[current_regime].append(duration_hours)
    
    # Convert to statistics
    stats = {}
    for regime, durs in durations.items():
        if len(durs) > 1:  # Need multiple observations
            stats[regime] = {
                "mean_duration": float(np.mean(durs)),
                "median_duration": float(np.median(durs)),
                "min_duration": float(np.min(durs)),
                "max_duration": float(np.max(durs)),
                "n_observations": len(durs)
            }
    
    return stats

def compute_transition_matrix(labels):
    """Compute regime transition probabilities"""
    unique_labels = sorted(np.unique(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    n_states = len(unique_labels)
    
    transition_matrix = np.zeros((n_states, n_states))
    for i in range(len(labels) - 1):
        current = label_to_idx[labels[i]]
        next_state = label_to_idx[labels[i+1]]
        transition_matrix[current, next_state] += 1
    
    # Normalize rows
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums!=0)
    
    # Convert to dict for JSON serialization
    transition_dict = {}
    for i, current_label in enumerate(unique_labels):
        transition_dict[str(current_label)] = {}
        for j, next_label in enumerate(unique_labels):
            if transition_matrix[i, j] > 0:
                transition_dict[str(current_label)][str(next_label)] = float(transition_matrix[i, j])
    
    return transition_dict

def merge_regime_stats(bias_stats, duration_stats):
    """Merge bias and duration statistics"""
    merged = {}
    all_regimes = set(bias_stats.keys()) | set(duration_stats.keys())
    
    for regime in all_regimes:
        merged[regime] = {}
        if regime in bias_stats:
            merged[regime].update(bias_stats[regime])
        if regime in duration_stats:
            merged[regime].update(duration_stats[regime])
    
    return merged

def save_regime_stats(regime_stats, path):
    """Save regime statistics to JSON"""
    with open(path, "w") as f:
        json.dump(regime_stats, f, indent=2)

def load_regime_stats(path):
    """Load regime statistics from JSON"""
    with open(path, "r") as f:
        return json.load(f)