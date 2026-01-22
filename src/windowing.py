import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

def create_windows_from_array(data: np.ndarray, window_size: int):
    """Create rolling windows from NumPy array"""
    X = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        X.append(window)
    return np.array(X, dtype=np.float32)

def create_normalized_windows(df: pd.DataFrame, window_size: int, use_global_scaling: bool = False, fit_scaler: bool = False, scaler_path: str = None):
    """Create normalized windows using raw NumPy arrays"""
    if use_global_scaling and fit_scaler:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df.values)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
    elif use_global_scaling and not fit_scaler:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        scaled_data = scaler.transform(df.values)
    else:
        scaled_data = df.values
    
    return create_windows_from_array(scaled_data, window_size)

def create_normalized_windows_multi(feature_dfs: dict, window_size: int, use_global_scaling: bool = True, fit_scaler: bool = False, scaler_path: str = None):
    """Create windows with global feature scaling using raw NumPy arrays"""
    all_windows = []
    asset_labels = []
    
    if use_global_scaling:
        # Fit global scaler on all assets (using raw arrays)
        all_data = np.concatenate([df.values for df in feature_dfs.values()], axis=0)
        scaler = StandardScaler()
        scaler.fit(all_data)
        
        # Save scaler if requested
        if fit_scaler and scaler_path:
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
        
        for asset, df in feature_dfs.items():
            scaled_data = scaler.transform(df.values)  # ← Critical: .values
            windows = create_windows_from_array(scaled_data, window_size)
            all_windows.append(windows)
            asset_labels.extend([asset] * len(windows))
    else:
        for asset, df in feature_dfs.items():
            windows = create_windows_from_array(df.values, window_size)
            all_windows.append(windows)
            asset_labels.extend([asset] * len(windows))
    
    return np.concatenate(all_windows, axis=0), np.array(asset_labels)

def get_aligned_forward_returns(close_series: pd.Series, window_size: int, horizon: int):
    """Get forward returns aligned with windows"""
    returns = close_series.pct_change(horizon).shift(-horizon)
    returns = returns.iloc[window_size - 1:-horizon]
    return returns.values.astype(np.float32)

def get_aligned_forward_returns_multi(asset_dfs: dict, window_size: int, horizon: int):
    """Get forward returns aligned with windows for multiple assets"""
    all_returns = []
    for asset, df in asset_dfs.items():
        returns = get_aligned_forward_returns(df["close"], window_size, horizon)
        all_returns.append(returns)
    return np.concatenate(all_returns, axis=0)