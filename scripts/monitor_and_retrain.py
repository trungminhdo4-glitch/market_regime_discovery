import os
import sys
import warnings
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion import load_ohlcv
from src.features import compute_features
from src.windowing import create_normalized_windows
from src.latent import extract_embeddings, load_conditioning, apply_conditioning
from src.clustering import load_cluster_model

UNSTABLE_REGIMES = {3, 9}
DRIFT_THRESHOLD = 0.05
WINDOW_HISTORY = 200
RETRAIN_LOCK_FILE = "artifacts/retrain.lock"
LOCK_DURATION_HOURS = 24

def is_retrain_locked():
    if not os.path.exists(RETRAIN_LOCK_FILE):
        return False
    with open(RETRAIN_LOCK_FILE, "r") as f:
        lock_time_str = f.read().strip()
    try:
        lock_time = datetime.fromisoformat(lock_time_str)
        return (datetime.now() - lock_time).total_seconds() < (LOCK_DURATION_HOURS * 3600)
    except:
        return False

def set_retrain_lock():
    with open(RETRAIN_LOCK_FILE, "w") as f:
        f.write(datetime.now().isoformat())

def compute_mmd(x, y, sigma=1.0):
    """Maximum Mean Discrepancy for latent drift detection"""
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    x_sqnorms = tf.reduce_sum(tf.square(x), axis=1)
    y_sqnorms = tf.reduce_sum(tf.square(y), axis=1)
    gamma = 1 / (2 * sigma**2)
    k_xx = tf.exp(-gamma * (x_sqnorms[:, None] + x_sqnorms[None, :] - 2 * tf.matmul(x, x, transpose_b=True)))
    k_yy = tf.exp(-gamma * (y_sqnorms[:, None] + y_sqnorms[None, :] - 2 * tf.matmul(y, y, transpose_b=True)))
    k_xy = tf.exp(-gamma * (x_sqnorms[:, None] + y_sqnorms[None, :] - 2 * tf.matmul(x, y, transpose_b=True)))
    mmd = tf.reduce_mean(k_xx) + tf.reduce_mean(k_yy) - 2 * tf.reduce_mean(k_xy)
    return float(mmd)

def get_recent_regime_assignments(df_raw: pd.DataFrame, window_size: int = 48):
    from tensorflow import keras
    encoder = keras.models.load_model("artifacts/encoder_only.keras")
    cluster_model = load_cluster_model("artifacts/cluster_model.pkl")
    conditioning = load_conditioning("artifacts/latent_conditioning.pkl")
    
    max_feature_lag = 60
    min_required = window_size + max_feature_lag + WINDOW_HISTORY
    if len(df_raw) < min_required:
        raise ValueError(f"Need at least {min_required} candles for monitoring")
    
    df_recent = df_raw.tail(min_required).copy()
    feature_config = {
        "log_return_lags": [1],
        "vol_windows": [5, 20, 60],
        "volume_zscore_window": 20,
        "add_vol_normalized_return": True
    }
    df_feat, _ = compute_features(df_recent, feature_config)
    
    X_windows = create_normalized_windows(
        df_feat, window_size,
        use_global_scaling=True,
        fit_scaler=False,
        scaler_path="artifacts/feature_scaler.pkl"
    )
    
    X_batch = X_windows[-WINDOW_HISTORY:]
    Z_batch = extract_embeddings(encoder, X_batch)
    Z_cond = apply_conditioning(Z_batch, conditioning)
    labels = cluster_model.predict(Z_cond)
    return labels, Z_cond

def should_retrain(labels: np.ndarray, Z_cond: np.ndarray, drift_config: dict) -> bool:
    # 1. Unstable regime frequency
    unstable_count = sum(1 for lbl in labels if lbl in UNSTABLE_REGIMES)
    unstable_ratio = unstable_count / len(labels)
    
    # 2. Latent distribution shift (MMD)
    baseline_Z = np.load("artifacts/baseline_latent.npy") if os.path.exists("artifacts/baseline_latent.npy") else Z_cond[:100]
    mmd_score = compute_mmd(baseline_Z, Z_cond)
    
    # 3. New regime emergence
    known_regimes = set(range(10))  # From training
    new_regimes = [lbl for lbl in labels if lbl not in known_regimes]
    new_regime_freq = len(new_regimes) / len(labels)
    
    print(f"📊 Monitoring Metrics:")
    print(f"   Unstable Regime Frequency: {unstable_ratio:.2%}")
    print(f"   MMD Score: {mmd_score:.3f}")
    print(f"   New Regime Frequency: {new_regime_freq:.2%}")
    
    return (
        unstable_ratio > drift_config["new_regime_threshold"] or
        mmd_score > drift_config["mmd_threshold"] or
        new_regime_freq > drift_config["new_regime_threshold"]
    )

def trigger_retrain():
    print("🔄 Starte Retraining...")
    os.system(f"{sys.executable} scripts/train.py")
    print("✅ Retraining abgeschlossen. Neue Artefakte verfügbar.")

def main():
    print("🔍 Prüfe auf Marktregime-Drift...")
    
    if is_retrain_locked():
        print("⏳ Retraining gesperrt (letztes Retraining vor <24h)")
        return
    
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    df_raw = load_ohlcv(
        config["data"]["path"],
        datetime_col=config["data"]["datetime_col"],
        tz=config["data"]["tz"]
    )
    print(f"   Geladen: {len(df_raw)} Candles (bis {df_raw.index[-1]})")
    
    try:
        labels, Z_cond = get_recent_regime_assignments(df_raw, config["windowing"]["size"])
    except Exception as e:
        print(f"❌ Fehler bei der Regime-Analyse: {e}")
        return
    
    if should_retrain(labels, Z_cond, config["drift_detection"]):
        print("⚠️  Drift erkannt! Triggerbedingungen erfüllt.")
        trigger_retrain()
        set_retrain_lock()
        # Save baseline for future MMD comparison
        np.save("artifacts/baseline_latent.npy", Z_cond)
    else:
        print("✅ Keine signifikante Drift erkannt. System stabil.")

if __name__ == "__main__":
    main()