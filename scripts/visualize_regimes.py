import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import matplotlib.pyplot as plt
import os

def visualize_regime_distribution():
    # 🔥 LOAD ASSETS FROM SAVED METADATA INSTEAD OF CONFIG
    try:
        asset_labels = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # Default assets
        # Try to load from actual saved data
        import numpy as np
        labels = np.load("artifacts/asset_labels.npy", allow_pickle=True)
        asset_labels = sorted(np.unique(labels).tolist())
    except:
        pass  # Use default assets
    
    # Plot vorbereiten
    fig, axes = plt.subplots(1, len(asset_labels), figsize=(5*len(asset_labels), 6))
    if len(asset_labels) == 1:
        axes = [axes]
    
    for i, asset in enumerate(asset_labels):
        try:
            with open(f"artifacts/regime_stats_{asset}.json") as f:
                stats = json.load(f)
            
            regimes = sorted(stats.keys(), key=int)
            n_samples = [stats[r]["n_samples"] for r in regimes]
            
            bars = axes[i].bar(regimes, n_samples, color='skyblue', edgecolor='navy')
            axes[i].set_title(f"{asset}", fontsize=12, fontweight='bold')
            axes[i].set_xlabel("Regime ID")
            axes[i].set_ylabel("Anzahl Samples")
            axes[i].grid(axis='y', alpha=0.3)
            
            # Werte auf Balken anzeigen
            for bar, count in zip(bars, n_samples):
                axes[i].text(
                    bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + max(n_samples)*0.01, 
                    f'{count}', 
                    ha='center', va='bottom', fontsize=9
                )
                
        except FileNotFoundError:
            axes[i].text(0.5, 0.5, f"Keine Daten\nfür {asset}", 
                        transform=axes[i].transAxes, ha='center', va='center')
    
    plt.suptitle("Regime-Verteilung pro Asset", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Speichern
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig("visualizations/regime_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Regime-Verteilung gespeichert unter: visualizations/regime_distribution.png")

if __name__ == "__main__":
    visualize_regime_distribution()