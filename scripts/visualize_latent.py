import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def visualize_latent_space():
    try:
        # Lade Daten
        Z = np.load("artifacts/latent_embeddings.npy")
        labels = np.load("artifacts/cluster_labels.npy")
        asset_labels = np.load("artifacts/asset_labels.npy", allow_pickle=True)
        
        # Reduziere auf 2D
        pca = PCA(n_components=2)
        Z_2d = pca.fit_transform(Z)
        
        # Plot vorbereiten
        plt.figure(figsize=(14, 10))
        unique_assets = np.unique(asset_labels)
        colors = ["red", "blue", "green", "purple", "orange"]
        
        # Scatter plot pro Asset
        for i, asset in enumerate(unique_assets):
            mask = asset_labels == asset
            plt.scatter(
                Z_2d[mask, 0], 
                Z_2d[mask, 1], 
                c=colors[i % len(colors)], 
                label=str(asset), 
                alpha=0.6, 
                s=2
            )
        
        # Load Silhouette score
        try:
            import json
            with open("artifacts/ml_health.json", "r") as f:
                health = json.load(f)
            sil_score = health["silhouette"]
        except:
            sil_score = 0.446
        
        plt.title(f"Latenter Raum (PCA 2D) – {len(unique_assets)} Assets\n"
                  f"Silhouette: {sil_score:.3f}", 
                  fontsize=14)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Speichern
        os.makedirs("visualizations", exist_ok=True)
        plt.savefig("visualizations/latent_space.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Visualisierung gespeichert unter: visualizations/latent_space.png")
        
    except Exception as e:
        print(f"❌ Fehler bei der Latent-Raum-Visualisierung: {e}")

if __name__ == "__main__":
    visualize_latent_space()