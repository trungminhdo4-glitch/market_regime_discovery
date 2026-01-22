import pandas as pd
import numpy as np

def compute_features(df: pd.DataFrame, config: dict):
    """Compute stationary features (without asset-specific logic)"""
    df = df.copy()
    close = df["close"]
    volume = df["volume"]

    # Log returns
    for lag in config["log_return_lags"]:
        df[f"log_ret_{lag}"] = np.log(close / close.shift(lag))

    # Multi-scale volatility
    log_ret = df["log_ret_1"]
    vol_windows = config.get("vol_windows", [5, 20])
    for win in vol_windows:
        df[f"vol_{win}"] = log_ret.rolling(window=win, min_periods=1).std()
    
    if len(vol_windows) >= 2:
        short_win = min(vol_windows)
        long_win = max(vol_windows)
        df["vol_ratio"] = df[f"vol_{short_win}"] / (df[f"vol_{long_win}"] + 1e-8)

    # Volume z-score
    vol_win = config["volume_zscore_window"]
    volume_ma = volume.rolling(window=vol_win, min_periods=1).mean()
    volume_std = volume.rolling(window=vol_win, min_periods=1).std()
    df["volume_z"] = (volume - volume_ma) / (volume_std + 1e-8)

    # Volatility-normalized return
    if config.get("add_vol_normalized_return", False):
        df["log_ret_norm"] = df["log_ret_1"] / (df[f"vol_{max(vol_windows)}"] + 1e-8)

    # Remove rows with too many NaNs
    max_window = max(
        vol_windows + [config["volume_zscore_window"]] + config["log_return_lags"]
    )
    df = df.iloc[max_window:].copy()

    return df

def compute_cross_asset_features(asset_dfs: dict):
    """Add cross-asset features to ALL assets with consistent columns"""
    if "BTCUSDT" not in asset_dfs:
        return asset_dfs
        
    assets = list(asset_dfs.keys())
    btc_close = asset_dfs["BTCUSDT"]["close"]
    btc_vol = asset_dfs["BTCUSDT"]["vol_20"] if "vol_20" in asset_dfs["BTCUSDT"] else None
    
    # 🔥 COLLECT ALL POSSIBLE FEATURES FIRST
    all_features = set()
    for asset in assets:
        all_features.update(asset_dfs[asset].columns.tolist())
    
    # Add cross-asset feature names
    for asset in assets:
        if asset != "BTCUSDT":
            all_features.add(f"{asset}_vs_btc")
            all_features.add(f"{asset}_btc_corr")
            if btc_vol is not None:
                all_features.add(f"{asset}_vol_spread")
    
    # Ensure BTC has placeholder features
    all_features.add("BTCUSDT_vs_btc")
    all_features.add("BTCUSDT_btc_corr")
    if btc_vol is not None:
        all_features.add("BTCUSDT_vol_spread")

    updated_dfs = {}
    for asset in assets:
        df = asset_dfs[asset].copy()
        
        # Add missing base features
        for feat in all_features:
            if feat not in df.columns:
                if feat in ["BTCUSDT_vs_btc", "BTCUSDT_btc_corr"]:
                    df[feat] = 1.0
                elif feat.endswith("_vs_btc") and feat.startswith(f"{asset}_"):
                    aligned_btc = btc_close.reindex(df.index, method='ffill')
                    df[feat] = df["close"] / aligned_btc
                elif feat.endswith("_btc_corr") and feat.startswith(f"{asset}_"):
                    asset_ret = df["close"].pct_change()
                    btc_ret = btc_close.pct_change().reindex(df.index, method='ffill')
                    df[feat] = (
                        asset_ret.rolling(20, min_periods=10)
                        .corr(btc_ret)
                    )
                elif feat.endswith("_vol_spread") and feat.startswith(f"{asset}_") and btc_vol is not None:
                    asset_vol = df["vol_20"] if "vol_20" in df else df["vol_ratio"]
                    btc_vol_aligned = btc_vol.reindex(df.index, method='ffill')
                    df[feat] = asset_vol - btc_vol_aligned
                else:
                    df[feat] = 0.0
        
        updated_dfs[asset] = df
    
    return updated_dfs

def compute_features_multi(asset_dfs: dict, config: dict):
    """Compute identical features for all assets"""
    feature_dfs = {}
    for asset, df in asset_dfs.items():
        df_feat = compute_features(df, config)
        feature_dfs[asset] = df_feat
    
    # Add cross-asset features
    feature_dfs = compute_cross_asset_features(feature_dfs)
    return feature_dfs