import os
import sys
import warnings
import yaml
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion import load_ohlcv_multi_with_onchain
from src.features import compute_features, compute_features_multi
from src.windowing import create_normalized_windows_multi, get_aligned_forward_returns_multi
from src.model import build_decoupled_autoencoder, temporal_loss
from src.latent import extract_embeddings, condition_latent_space_umap, save_conditioning
from src.clustering import fit_clustering_model, save_cluster_model
from src.regimes import compute_regime_bias, compute_regime_durations, merge_regime_stats, save_regime_stats, compute_transition_matrix

def add_noise(x_clean, noise_config):
    noise_type = noise_config["noise_type"]
    intensity = noise_config["noise_intensity"]
    if noise_type == "gaussian":
        return x_clean + tf.random.normal(tf.shape(x_clean), stddev=intensity)
    return x_clean

def add_feature_dropout(x_clean, dropout_rate=0.1):
    if tf.random.uniform(()) < dropout_rate:
        feature_idx = tf.random.uniform((), maxval=tf.shape(x_clean)[-1], dtype=tf.int32)
        mask = tf.ones(tf.shape(x_clean))
        mask = tf.concat([
            mask[:, :, :feature_idx],
            tf.zeros_like(mask[:, :, :1]),
            mask[:, :, feature_idx+1:]
        ], axis=-1)
        return x_clean * mask
    return x_clean

def compute_health_metrics(Z, labels, X, X_recon):
    metrics = {}
    metrics["latent_variance"] = float(np.var(Z, axis=0).mean())
    metrics["recon_loss"] = float(np.mean((X - X_recon)**2))
    metrics["silhouette"] = float(silhouette_score(Z, labels))
    
    durations = []
    current_regime = labels[0]
    count = 1
    for label in labels[1:]:
        if label == current_regime:
            count += 1
        else:
            durations.append(count)
            current_regime = label
            count = 1
    metrics["regime_persistence"] = float(np.mean(durations)) if durations else 0.0
    
    transitions = np.diff(labels)
    unique, counts = np.unique(transitions, return_counts=True)
    probs = counts / len(transitions)
    entropy_val = -np.sum(probs * np.log(probs + 1e-8))
    metrics["transition_entropy"] = float(entropy_val)
    
    return metrics

def get_window_timestamps(asset_dfs: dict, window_size: int, asset_labels: np.ndarray):
    """Get timestamps corresponding to each window"""
    timestamps = []
    current_idx = 0
    
    for asset, df in asset_dfs.items():
        asset_mask = asset_labels == asset
        asset_count = np.sum(asset_mask)
        
        if asset_count > 0:
            # Get valid timestamps for this asset (after feature computation lag)
            max_feature_lag = 60  # From features.py
            valid_df = df.iloc[max_feature_lag:].copy()
            valid_timestamps = valid_df.index[window_size - 1:]  # Align with windows
            
            # Take only the timestamps we need
            needed_timestamps = valid_timestamps[:asset_count]
            timestamps.extend(needed_timestamps)
            current_idx += asset_count
    
    return pd.DatetimeIndex(timestamps)

def main():
    config_path = "config/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tf.keras.utils.set_random_seed(config["training"]["random_seed"])
    tf.config.experimental.enable_op_determinism()

    assets = config["assets"]
    asset_dfs = load_ohlcv_multi_with_onchain(assets, config["data"]["base_path"])
    print(f"Geladen: {len(assets)} Assets mit gemeinsamem Zeitraum")

    # 🔥 REMOVE INCONSISTENT ON-CHAIN COLUMNS IF NOT AVAILABLE FOR ALL ASSETS
    onchain_cols = ["active_addresses", "gas_used"]
    cols_to_remove = []
    for col in onchain_cols:
        has_col = [col in asset_dfs[asset].columns for asset in assets]
        if not all(has_col):
            cols_to_remove.append(col)
            print(f"⚠️  Entferne On-Chain-Spalte '{col}' (nicht für alle Assets verfügbar)")
    
    if cols_to_remove:
        for asset in assets:
            asset_dfs[asset] = asset_dfs[asset].drop(columns=cols_to_remove, errors='ignore')

    feature_dfs = compute_features_multi(asset_dfs, config["features"])
    print(f"Features berechnet für alle Assets (inkl. Cross-Asset)")

    sample_asset = assets[0]
    sample_df = feature_dfs[sample_asset]
    all_feature_cols = sorted(sample_df.columns.tolist())

    shape_features = [col for col in all_feature_cols if "vol_ratio" in col or "log_ret_norm" in col]
    energy_features = [col for col in all_feature_cols if col not in shape_features]

    feature_groups = {
        "shape_cols": shape_features,
        "energy_cols": energy_features,
        "all_cols": all_feature_cols
    }

    window_size = config["windowing"]["size"]
    X, asset_labels = create_normalized_windows_multi(
        feature_dfs, 
        window_size, 
        use_global_scaling=config["windowing"].get("use_global_scaling", True),
        fit_scaler=True,
        scaler_path="artifacts/feature_scaler.pkl"
    )

    print(f"Vor NaN-Handling: {np.isnan(X).sum()} NaN-Werte")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"Nach NaN-Handling: {np.isnan(X).sum()} NaN-Werte")

    print(f"Windows erstellt: {X.shape} ({len(np.unique(asset_labels))} Assets)")

    forward_horizon = config["regimes"]["forward_horizon"]
    forward_rets = get_aligned_forward_returns_multi(asset_dfs, window_size, forward_horizon)
    
    min_len = min(len(X), len(forward_rets))
    if len(X) != len(forward_rets):
        print(f"⚠️  Längen ungleich – kürze auf {min_len} Samples")
        X = X[:min_len]
        forward_rets = forward_rets[:min_len]
        asset_labels = asset_labels[:min_len]
    print(f"✅ Endgültige Datenform: X={X.shape}, forward_rets={forward_rets.shape}")

    # 🔥 GET WINDOW TIMESTAMPS FOR DURATION CALCULATION
    window_timestamps = get_window_timestamps(asset_dfs, window_size, asset_labels)
    if len(window_timestamps) != len(X):
        print(f"⚠️  Timestamp mismatch: {len(window_timestamps)} vs {len(X)}")
        window_timestamps = window_timestamps[:len(X)]

    model_cfg = config["model"]
    autoencoder = build_decoupled_autoencoder(
        input_shape=(window_size, X.shape[2]),
        latent_dim=model_cfg["latent_dim"],
        feature_groups=feature_groups,
        temporal_lambda=model_cfg["temporal_consistency_lambda"],
        multi_horizon_lambda=model_cfg["multi_horizon_lambda"],
        contrastive_weight=model_cfg["contrastive_weight"]
    )

    train_cfg = config["training"]
    optimizer = tf.keras.optimizers.Adam()
    
    @tf.function
    def train_step(x_clean):
        x_noisy = add_noise(x_clean, train_cfg)
        x_noisy = add_feature_dropout(x_noisy, train_cfg["feature_dropout_rate"])
        with tf.GradientTape() as tape:
            recon = autoencoder(x_noisy, training=True)
            z = autoencoder.encoder(x_noisy, training=True)
            loss = temporal_loss(autoencoder, x_clean, recon, z)
        grads = tape.gradient(loss, autoencoder.trainable_weights)
        optimizer.apply_gradients(zip(grads, autoencoder.trainable_weights))
        return loss

    print("🚀 Starte Multi-Asset Training mit Cross-Asset Features...")
    batch_size = train_cfg["batch_size"]
    for epoch in range(train_cfg["epochs"]):
        epoch_loss = 0.0
        num_batches = 0
        for i in range(0, len(X) - batch_size + 1, batch_size):
            batch = X[i:i + batch_size]
            loss = train_step(batch)
            epoch_loss += loss
            num_batches += 1
        
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/{train_cfg['epochs']} - Loss: {avg_loss:.4f}")

    Z = extract_embeddings(autoencoder.encoder, X)
    
    cond_cfg = config["clustering"]["latent_conditioning"]
    if cond_cfg.get("use_umap", False):
        Z_cond, conditioning = condition_latent_space_umap(
            Z, 
            n_components=cond_cfg.get("umap_components", 4)
        )
    else:
        from src.latent import condition_latent_space
        Z_cond, conditioning = condition_latent_space(Z, cond_cfg["variance_threshold"])
    
    save_conditioning(conditioning, "artifacts/latent_conditioning.pkl")
    print(f"📉 Latent dimensionality reduced to {Z_cond.shape[1]} ({conditioning['method']})")

    cluster_cfg = config["clustering"]
    cluster_model, labels = fit_clustering_model(
        Z_cond, 
        method=cluster_cfg["method"], 
        config=cluster_cfg,
        random_state=cluster_cfg["random_state"]
    )
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"🎯 GMM wählte {len(unique_labels)} Cluster")

    # 🔥 COMPUTE ASSET-SPECIFIC REGIME STATISTICS WITH DURATIONS
    regime_stats_multi = {}
    for asset in assets:
        asset_mask = asset_labels == asset
        asset_regimes = labels[asset_mask]
        asset_returns = forward_rets[asset_mask]
        asset_timestamps = window_timestamps[asset_mask]
        
        # Compute bias statistics
        bias_stats = compute_regime_bias(asset_regimes, asset_returns, config.get("regimes", {}))
        
        # Compute duration statistics
        duration_stats = {}
        if len(asset_regimes) > 1:
            try:
                duration_stats = compute_regime_durations(
                    asset_regimes, 
                    asset_timestamps, 
                    window_size=window_size
                )
            except Exception as e:
                print(f"⚠️  Fehler bei Dauerberechnung für {asset}: {e}")
        
        # Merge statistics
        regime_stats_multi[asset] = merge_regime_stats(bias_stats, duration_stats)

    # 🔥 COMPUTE TRANSITION MATRIX
    transition_matrix = compute_transition_matrix(labels)
    with open("artifacts/transition_matrix.json", "w") as f:
        json.dump(transition_matrix, f, indent=2)
    print("📊 Transition Matrix gespeichert")

    X_recon = autoencoder.predict(X, verbose=0)
    health_metrics = compute_health_metrics(Z_cond, labels, X, X_recon)
    print(f"📊 Silhouette: {health_metrics['silhouette']:.3f} | Transition Entropy: {health_metrics['transition_entropy']:.3f}")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    autoencoder.save("checkpoints/autoencoder.keras")
    autoencoder.encoder.save("artifacts/encoder_only.keras")
    autoencoder.save_weights("checkpoints/autoencoder.weights.h5")
    
    np.save("artifacts/latent_embeddings.npy", Z_cond)
    np.save("artifacts/cluster_labels.npy", labels)
    np.save("artifacts/asset_labels.npy", asset_labels)
    
    with open("artifacts/feature_groups.json", "w") as f:
        json.dump(feature_groups, f)
    
    save_cluster_model(cluster_model, "artifacts/cluster_model.pkl")
    
    for asset, stats in regime_stats_multi.items():
        save_regime_stats(stats, f"artifacts/regime_stats_{asset}.json")
    
    with open("artifacts/ml_health.json", "w") as f:
        json.dump(health_metrics, f, indent=2)

    print("\n✅ Multi-Asset Training abgeschlossen. Artefakte gespeichert.")
    for asset in assets:
        print(f"\n{asset} Regime-Statistiken:")
        for k, v in sorted(regime_stats_multi[asset].items()):
            duration_info = ""
            if "mean_duration" in v:
                duration_info = f" | Dauer: {v['mean_duration']:.1f}h"
            vol_context = ""
            if "volatility_regime" in v and "volatility_market" in v:
                vol_ratio = v["volatility_regime"] / (v["volatility_market"] + 1e-8)
                if abs(vol_ratio - 1.0) > 0.1:
                    vol_context = f" | Vol {'↑' if vol_ratio > 1 else '↓'}{abs(vol_ratio - 1):.0%}"
            print(f"  Regime {k}: P(up) = {v['p_up']:.2%}, Sharpe = {v['sharpe']:.2f}, N = {v['n_samples']}{duration_info}{vol_context}")

if __name__ == "__main__":
    main()