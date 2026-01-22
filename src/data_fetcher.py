import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np

class BinanceOHLCVFetcher:
    # 🔥 FIXED BASE URL (removed trailing spaces + corrected path)
    BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

    def __init__(self, symbol: str = "BTCUSDT", interval: str = "1h"):
        self.symbol = symbol
        self.interval = interval

    def _generate_url(self, year: int, month: int) -> str:
        month_str = f"{month:02d}"
        filename = f"{self.symbol}-{self.interval}-{year}-{month_str}.zip"
        # 🔥 CORRECTED URL STRUCTURE
        return f"{self.BASE_URL}/{self.symbol}/{self.interval}/{filename}"

    def _fetch_month(self, year: int, month: int) -> Optional[pd.DataFrame]:
        url = self._generate_url(year, month)
        try:
            print(f"📡 Requesting: {url}")  # Debug line
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                df = pd.read_csv(
                    pd.io.common.BytesIO(response.content),
                    compression="zip",
                    header=None,
                    dtype=float
                )
                return self._format_dataframe(df)
            else:
                print(f"⚠️  Keine Daten für {year}-{month:02d} (HTTP {response.status_code})")
                return None
        except Exception as e:
            print(f"❌ Fehler beim Laden von {year}-{month:02d}: {e}")
            return None

    def _format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.iloc[:, :6].copy()
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        return df[["open", "high", "low", "close", "volume"]]

    def fetch_range(self, start_date: str, end_date: str = None) -> pd.DataFrame:
        start_naive = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        start_utc = pd.Timestamp(start_naive).tz_localize("UTC")

        if end_date:
            end_naive = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            end_utc = pd.Timestamp(end_naive).tz_localize("UTC")
        else:
            end_utc = pd.Timestamp.utcnow().tz_convert("UTC")

        all_dfs = []
        current = pd.Timestamp(year=start_utc.year, month=start_utc.month, day=1, tz="UTC")

        while current <= end_utc:
            year, month = current.year, current.month
            print(f"📥 Lade {self.symbol} {self.interval} für {year}-{month:02d}...")
            df_month = self._fetch_month(year, month)
            if df_month is not None:
                month_start = current
                month_end = pd.Timestamp(
                    year=year + (1 if month == 12 else 0),
                    month=1 if month == 12 else month + 1,
                    day=1, tz="UTC"
                )
                mask = (df_month.index >= month_start) & (df_month.index < month_end)
                df_filtered = df_month[mask]
                if not df_filtered.empty:
                    all_dfs.append(df_filtered)

            if current.month == 12:
                current = pd.Timestamp(year=current.year + 1, month=1, day=1, tz="UTC")
            else:
                current = pd.Timestamp(year=current.year, month=current.month + 1, day=1, tz="UTC")

        if not all_dfs:
            raise ValueError("Keine Daten im angegebenen Zeitraum gefunden.")

        full_df = pd.concat(all_dfs).sort_index()
        full_df = full_df[~full_df.index.duplicated(keep="first")]
        return full_df

    def save_to_csv(self, df: pd.DataFrame, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.index = df.index.tz_localize(None)
        df.to_csv(path, encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S")
        print(f"✅ Daten gespeichert unter: {path}")

    
    # Am Ende der Datei hinzufügen
def fetch_onchain_data(asset: str, start_date: str, end_date: str = None):
    """Lade On-Chain-Daten von öffentlichen APIs"""
    if asset == "BTCUSDT":
        # Glassnode Active Addresses (vereinfacht)
        url = f"https://api.glassnode.com/v1/metrics/addresses/active_count?a=BTC&since={start_date}&until={end_date or '2024-12-31'}"
        # In der Praxis benötigst du einen API-Key
        # Für Demo: Nutze synthetische Daten
        return pd.Series(np.random.normal(1e6, 1e5, 1000), 
                        index=pd.date_range(start_date, periods=1000, freq='D'))
    
    elif asset == "ETHUSDT":
        # Etherscan Gas Used (vereinfacht)
        url = f"https://api.etherscan.io/api?module=stats&action=ethsupply&apikey=YourApiKey"
        return pd.Series(np.random.normal(1e7, 1e6, 1000),
                        index=pd.date_range(start_date, periods=1000, freq='D'))
    
    return None

def save_onchain_data(data: pd.Series, path: str):
    """Speichere On-Chain-Daten als CSV"""
    data.to_csv(path, header=True)