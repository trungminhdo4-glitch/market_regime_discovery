import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_fetcher import BinanceOHLCVFetcher, fetch_onchain_data, save_onchain_data

def main():
    ASSETS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    INTERVAL = "1h"
    START_DATE = "2020-01-01"
    
    # OHLCV-Daten
    for symbol in ASSETS:
        print(f"\n📥 Lade {symbol} OHLCV...")
        fetcher = BinanceOHLCVFetcher(symbol=symbol, interval=INTERVAL)
        df = fetcher.fetch_range(start_date=START_DATE)
        fetcher.save_to_csv(df, f"data/raw/{symbol}_1h.csv")
    
    # On-Chain-Daten
    print("\n📥 Lade On-Chain-Daten...")
    for asset in ["BTCUSDT", "ETHUSDT"]:
        onchain_data = fetch_onchain_data(asset, START_DATE)
        if onchain_data is not None:
            save_onchain_data(onchain_data, f"data/raw/{asset}_onchain.csv")
            print(f"✅ On-Chain-Daten gespeichert für {asset}")

if __name__ == "__main__":
    main()