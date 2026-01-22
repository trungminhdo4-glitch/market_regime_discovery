import os
import pandas as pd

def load_ohlcv(asset: str, base_path: str = "data/raw"):
    """Lade OHLCV-Daten für ein Asset"""
    # 🔥 KORRIGIERTE PFAD-LOGIK
    csv_path = os.path.join(base_path, f"{asset}_1h.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV-Datei nicht gefunden: {csv_path}")
    
    df = pd.read_csv(csv_path, encoding="utf-8")
    if "timestamp" not in df.columns:
        raise ValueError("CSV muss eine 'timestamp'-Spalte enthalten.")
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return df

def load_onchain_data(asset: str, base_path: str = "data/raw"):
    """Lade On-Chain-Daten für Asset"""
    csv_path = os.path.join(base_path, f"{asset}_onchain.csv")
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = df.index.tz_localize("UTC")
    return df.squeeze()

def load_ohlcv_multi_with_onchain(assets, base_path="data/raw"):
    """Lade OHLCV + On-Chain-Daten"""
    asset_dfs = {}
    for asset in assets:
        # 🔥 Lade OHLCV mit korrektem Pfad
        ohlcv = load_ohlcv(asset, base_path)
        onchain = load_onchain_data(asset, base_path)
        
        if onchain is not None:
            # Resample On-Chain auf stündlich
            onchain_hourly = onchain.resample('H').ffill()
            # Merge mit OHLCV
            merged = ohlcv.join(onchain_hourly, how='left')
            merged.iloc[:, -1] = merged.iloc[:, -1].fillna(method='ffill')
            asset_dfs[asset] = merged
        else:
            asset_dfs[asset] = ohlcv
    
    # Align all assets to common time period
    common_index = asset_dfs[assets[0]].index
    for asset in assets[1:]:
        common_index = common_index.intersection(asset_dfs[asset].index)
    
    aligned_dfs = {}
    for asset in assets:
        aligned_dfs[asset] = asset_dfs[asset].loc[common_index]
    
    return aligned_dfs