import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def show_model_architecture():
    try:
        from tensorflow import keras
        from src.model import build_decoupled_autoencoder
        import yaml
        import json
        import os
        
        # Load config and feature groups
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        with open("artifacts/feature_groups.json", "r", encoding="utf-8") as f:
            feature_groups = json.load(f)
        
        # Rebuild model
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
        
        print("\n" + "="*80)
        print("MODEL ARCHITECTURE SUMMARY")
        print("="*80)
        autoencoder.summary()
        print("="*80)
        
        # Speichere als Textdatei mit UTF-8 Kodierung
        os.makedirs("visualizations", exist_ok=True)
        with open("visualizations/model_summary.txt", "w", encoding="utf-8") as f:
            autoencoder.summary(print_fn=lambda x: f.write(x + '\n'))
        print("\n✅ Modellzusammenfassung gespeichert unter: visualizations/model_summary.txt")
        
    except Exception as e:
        print(f"❌ Fehler beim Rekonstruieren des Modells: {e}")

if __name__ == "__main__":
    show_model_architecture()