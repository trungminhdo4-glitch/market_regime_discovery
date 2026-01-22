import sys
import os
import warnings
import pandas as pd
import numpy as np
import argparse
import json
import pickle
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="keras")
warnings.filterwarnings("ignore", message="infer_datetime_format")

sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion import load_ohlcv
from src.features import compute_features
from src.latent import extract_embeddings, load_conditioning, apply_conditioning
from src.clustering import load_cluster_model
from src.regimes import load_regime_stats

UNSTABLE_REGIMES = {5, 7, 9}

def load_artifacts(asset: str):
    try:
        from tensorflow import keras
        from src.model import build_decoupled_autoencoder
        import yaml
        
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

def load_latest_ohlcv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV-Datei nicht gefunden: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8")
    if "timestamp" not in df.columns:
        raise ValueError("CSV muss eine 'timestamp'-Spalte enthalten.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    required_cols = ["open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Fehlende Spalten. Erwartet: {required_cols}")

    df = df[required_cols].astype(float)
    return df

def infer_latest_ohlcv(df_raw: pd.DataFrame, window_size: int = 48, target_asset: str = "BTCUSDT"):
    max_feature_lag = 60
    min_required_raw = window_size + max_feature_lag

    if len(df_raw) < min_required_raw:
        raise ValueError(
            f"Nicht genug Candles für Inferenz. "
            f"Benötigt: {min_required_raw}, Vorhanden: {len(df_raw)}"
        )

    df_recent = df_raw.tail(min_required_raw).copy()
    
    # Compute base features (NUR 2 PARAMETER!)
    feature_config = {
        "log_return_lags": [1],
        "vol_windows": [5, 20, 60],
        "volume_zscore_window": 20,
        "add_vol_normalized_return": True
    }
    df_feat = compute_features(df_recent, feature_config)  # ← Korrigiert: Kein target_asset
    
    # Add missing cross-asset features
    with open("artifacts/feature_groups.json", "r") as f:
        feature_groups = json.load(f)
    all_features = feature_groups["all_cols"]

    for col in all_features:
        if col not in df_feat.columns:
            if target_asset == "BTCUSDT":
                # BTC gets placeholders
                if col.endswith("_vs_btc"):
                    df_feat[col] = 1.0
                elif col.endswith("_btc_corr"):
                    df_feat[col] = 1.0
                elif col.endswith("_vol_spread"):
                    df_feat[col] = 0.0
                else:
                    df_feat[col] = 0.0
            else:
                # Non-BTC assets get real values for their own features, 0 otherwise
                if col.startswith(f"{target_asset}_"):
                    if "_vs_btc" in col:
                        btc_path = "data/raw/BTCUSDT_1h.csv"
                        if os.path.exists(btc_path):
                            btc_df = pd.read_csv(btc_path, encoding="utf-8")
                            btc_df["timestamp"] = pd.to_datetime(btc_df["timestamp"], utc=True)
                            btc_df = btc_df.set_index("timestamp").sort_index()
                            btc_close = btc_df["close"].reindex(df_feat.index, method='ffill')
                            df_feat[col] = df_feat["close"] / btc_close
                        else:
                            df_feat[col] = 1.0  # Fallback
                    elif "_btc_corr" in col:
                        df_feat[col] = 0.0  # Can't compute correlation without full history
                    elif "_vol_spread" in col:
                        df_feat[col] = 0.0  # Can't compute without BTC vol
                    else:
                        df_feat[col] = 0.0
                else:
                    df_feat[col] = 0.0

    # Enforce exact column order from training
    df_feat = df_feat[all_features]
    
    if len(df_feat) < window_size:
        raise RuntimeError(
            f"Nach Feature-Berechnung zu wenige Zeilen: {len(df_feat)} < {window_size}."
        )

    # 🔥 USE RAW NUMPY ARRAY FOR SCALING (BYPASS FEATURE NAMES)
    with open("artifacts/feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    scaled_data = scaler.transform(df_feat.values)  # ← Critical: .values
    
    # Create single window from last window_size rows
    start_idx = len(scaled_data) - window_size
    window = scaled_data[start_idx:start_idx + window_size]
    return np.array([window], dtype=np.float32)

def format_regime_context(stats):
    """Format complete regime context"""
    parts = []
    
    # Duration
    if "mean_duration" in stats:
        parts.append(f"⌀ {stats['mean_duration']:.1f}h")
    
    # Volatility context
    if "volatility_regime" in stats and "volatility_market" in stats:
        vol_ratio = stats["volatility_regime"] / (stats["volatility_market"] + 1e-8)
        if abs(vol_ratio - 1.0) > 0.1:  # Nur wenn signifikant unterschiedlich
            parts.append(f"Vol {'↑' if vol_ratio > 1 else '↓'}{abs(vol_ratio - 1):.0%}")
    
    return f" ({', '.join(parts)})" if parts else ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=str, default="BTCUSDT", help="Asset symbol (e.g., BTCUSDT, ETHUSDT)")
    args = parser.parse_args()
    
    asset = args.asset
    print(f"🔍 Lade trainierte Artefakte für {asset}...")
    try:
        encoder, cluster_model, regime_stats, conditioning = load_artifacts(asset)
    except RuntimeError as e:
        print(f"❌ {e}")
        return

    csv_path = f"data/raw/{asset}_1h.csv"
    print(f"📥 Lade aktuelle Marktdaten für {asset}...")
    try:
        df_raw = load_latest_ohlcv(csv_path)
    except Exception as e:
        print(f"❌ {e}")
        return

    print(f"   Geladen: {len(df_raw)} Candles (bis {df_raw.index[-1]})")

    print("⚙️  Bereite Daten für Inferenz vor...")
    try:
        X_input = infer_latest_ohlcv(df_raw, window_size=48, target_asset=asset)
    except Exception as e:
        print(f"❌ {e}")
        return

    print("🧠 Führe Inferenz durch...")
    try:
        Z = extract_embeddings(encoder, X_input)
        Z_cond = apply_conditioning(Z, conditioning)
        cluster_id = int(cluster_model.predict(Z_cond)[0])
        stats = regime_stats.get(str(cluster_id))
        
        # 🔥 ROBUSTER FALLBACK FÜR FEHLENDE REGIME
        if stats is None:
            available_regimes = list(regime_stats.keys())
            if available_regimes:
                # Nimm das häufigste Regime als Fallback
                fallback_regime = max(available_regimes, key=lambda r: regime_stats[r]["n_samples"])
                stats = regime_stats[fallback_regime]
                print(f"⚠️  Regime {cluster_id} nicht verfügbar für {asset}. Nutze Fallback: Regime {fallback_regime}")
            else:
                raise ValueError(f"Keine Regime-Statistiken für {asset} gefunden.")
            
        p_up = stats["p_up"]
        n_samples = stats["n_samples"]
        context_info = format_regime_context(stats)
    except Exception as e:
        print(f"❌ Inferenz-Fehler: {e}")
        return

    stability_msg = ""
    if cluster_id in UNSTABLE_REGIMES:
        stability_msg = (
            "\n⚠️  ACHTUNG: Dieses Regime zeigte signifikante Drift in der "
            "Walk-Forward-Validierung (ΔP(up) > 3%).\n"
            "   Interpretieren Sie das Signal mit besonderer Vorsicht!"
        )

    print(f"\n✅ Aktuelles Marktregime für {asset} erkannt:")
    print(f"   Regime-ID       : {cluster_id}")
    print(f"   P(up next hour) : {p_up:.2%}{context_info}")
    print(f"   Historische Basis: {n_samples} Beispiele")
    print(f"   Letzter Candle  : {df_raw.index[-1]}")
    if stability_msg:
        print(stability_msg)

if __name__ == "__main__":
    main()