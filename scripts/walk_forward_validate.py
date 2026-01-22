import os
import sys
import warnings
import yaml
import json
import numpy as np
import pandas as pd
import argparse
import pickle
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion import load_ohlcv
from src.features import compute_features
from src.latent import extract_embeddings, load_conditioning, apply_conditioning
from src.clustering import load_cluster_model
from src.regimes import load_regime_stats

def load_artifacts(asset: str):
    """Lade trainierte Artefakte für Walk-Forward-Validierung"""
    try:
        from tensorflow import keras
        from src.model import build_decoupled_autoencoder
        
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        with open("artifacts/feature_groups.json", "r", encoding="utf-8") as f:
            feature_groups = json.load(f)
        
        window_size = config["windowing"]["size"]
        n_features = len(feature_groups["all_cols"])
        model_cfg = config["model"]
        
        autoencoder = build_decoupled_autoencoder(
            input_shape=(window_size, n_features),
            latent_dim=model_cfg["latent_dim"],
            feature_groups=feature_groups,
            temporal_lambda=model_cfg.get("temporal_consistency_lambda", 0.0),
            multi_horizon_lambda=model_cfg.get("multi_horizon_lambda", 0.0),
            contrastive_weight=model_cfg.get("contrastive_weight", 0.0)
        )
        
        autoencoder.load_weights("checkpoints/autoencoder.weights.h5")
        encoder = autoencoder.encoder
        
    except Exception as e:
        raise RuntimeError(f"Encoder konnte nicht geladen werden: {e}")

    try:
        cluster_model = load_cluster_model("artifacts/cluster_model.pkl")
    except Exception as e:
        raise RuntimeError(f"Cluster-Modell konnte nicht geladen werden: {e}")

    try:
        regime_stats = load_regime_stats(f"artifacts/regime_stats_{asset}.json")
    except Exception as e:
        raise RuntimeError(f"Regime-Statistiken für {asset} konnten nicht geladen werden: {e}")

    try:
        conditioning = load_conditioning("artifacts/latent_conditioning.pkl")
    except Exception as e:
        raise RuntimeError(f"Latent Conditioning konnte nicht geladen werden: {e}")

    return encoder, cluster_model, regime_stats, conditioning

def validate_regime_stability(asset: str, df: pd.DataFrame, window_size: int = 48):
    """Führe Walk-Forward-Validierung durch"""
    print(f"⚙️  Führe Walk-Forward-Validierung für {asset} durch...")
    
    # Lade Artefakte
    encoder, cluster_model, regime_stats, conditioning = load_artifacts(asset)
    
    # Berechne Rolling Windows (mit ausreichendem Puffer für Features)
    max_feature_lag = 60  # Aus features.py
    min_required_window = window_size + max_feature_lag
    
    if len(df) < min_required_window:
        raise ValueError(f"Datensatz zu klein für Validierung. Benötigt: {min_required_window}, Vorhanden: {len(df)}")
    
    total_valid_windows = len(df) - min_required_window + 1
    step_size = max(1, total_valid_windows // 50)  # Max 50 Validierungspunkte
    
    drift_points = []
    unstable_regimes = set()
    
    for i in range(0, total_valid_windows, step_size):
        # Extrahiere ausreichend große Daten für Features
        end_idx = i + min_required_window
        window_df = df.iloc[i:end_idx].copy()
        
        # Bereite Features vor
        feature_config = {
            "log_return_lags": [1],
            "vol_windows": [5, 20, 60],
            "volume_zscore_window": 20,
            "add_vol_normalized_return": True
        }
        df_feat = compute_features(window_df, feature_config)
        
        # Sicherstellen, dass genug Daten für das Fenster vorhanden sind
        if len(df_feat) < window_size:
            continue  # Überspringe zu kleine Fenster
        
        # Nimm nur das letzte Fenster
        df_feat = df_feat.tail(window_size).copy()
        
        # Füge fehlende Cross-Asset-Features hinzu
        with open("artifacts/feature_groups.json", "r") as f:
            feature_groups = json.load(f)
        all_features = feature_groups["all_cols"]
        
        for col in all_features:
            if col not in df_feat.columns:
                if asset == "BTCUSDT":
                    if col.endswith("_vs_btc"):
                        df_feat[col] = 1.0
                    elif col.endswith("_btc_corr"):
                        df_feat[col] = 1.0
                    elif col.endswith("_vol_spread"):
                        df_feat[col] = 0.0
                    else:
                        df_feat[col] = 0.0
                else:
                    if col.startswith(f"{asset}_"):
                        if "_vs_btc" in col:
                            btc_path = "data/raw/BTCUSDT_1h.csv"
                            if os.path.exists(btc_path):
                                btc_df = pd.read_csv(btc_path, encoding="utf-8")
                                btc_df["timestamp"] = pd.to_datetime(btc_df["timestamp"], utc=True)
                                btc_df = btc_df.set_index("timestamp").sort_index()
                                btc_close = btc_df["close"].reindex(df_feat.index, method='ffill')
                                df_feat[col] = df_feat["close"] / btc_close
                            else:
                                df_feat[col] = 1.0
                        elif "_btc_corr" in col:
                            df_feat[col] = 0.0
                        elif "_vol_spread" in col:
                            df_feat[col] = 0.0
                        else:
                            df_feat[col] = 0.0
                    else:
                        df_feat[col] = 0.0
        
        # Erzwinge Feature-Reihenfolge
        df_feat = df_feat[all_features]
        
        # Sicherstellen, dass das Array nicht leer ist
        if len(df_feat) == 0:
            continue
            
        # Skalierung
        with open("artifacts/feature_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        scaled_data = scaler.transform(df_feat.values)
        X_input = np.array([scaled_data], dtype=np.float32)
        
        # Inferenz
        Z = extract_embeddings(encoder, X_input)
        Z_cond = apply_conditioning(Z, conditioning)
        cluster_id = int(cluster_model.predict(Z_cond)[0])
        
        # Validierung
        stats = regime_stats.get(str(cluster_id))
        if stats:
            current_p_up = stats["p_up"]
            # Prüfe auf Drift (vereinfacht)
            if abs(current_p_up - 0.5) < 0.01:  # Sehr nahe an 50%
                drift_points.append((i, cluster_id))
                unstable_regimes.add(cluster_id)
    
    return list(unstable_regimes), drift_points

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=str, required=True, help="Asset symbol (e.g., BTCUSDT)")
    args = parser.parse_args()

    print("🔍 Lade Konfiguration...")
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # KORRIGIERTE DATENPFAD-KONFIGURATION
    data_path = config["data"]["base_path"]
    
    print(f"📥 Lade historische Daten für {args.asset}...")
    df = load_ohlcv(args.asset, data_path)
    print(f"   Geladen: {len(df)} Candles")
    
    window_size = config["windowing"]["size"]
    unstable_regimes, drift_points = validate_regime_stability(args.asset, df, window_size)
    
    print(f"\n📊 Walk-Forward-Validierung abgeschlossen für {args.asset}:")
    print(f"   Instabile Regime identifiziert: {sorted(list(unstable_regimes)) if unstable_regimes else 'Keine'}")
    print(f"   Drift-Punkte erkannt: {len(drift_points)}")
    
    if unstable_regimes:
        print(f"\n⚠️  WARNUNG: Folgende Regime zeigten Instabilität:")
        for regime in sorted(unstable_regimes):
            print(f"   - Regime {regime}")
    else:
        print(f"\n✅ Alle Regime stabil – kein Retraining erforderlich.")

if __name__ == "__main__":
    main()