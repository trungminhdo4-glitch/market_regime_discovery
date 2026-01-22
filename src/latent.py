import pickle
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP

def extract_embeddings(encoder, X, batch_size=256):
    embeddings = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        emb = encoder.predict(batch, verbose=0)
        embeddings.append(emb)
    return np.concatenate(embeddings, axis=0)

def condition_latent_space(Z: np.ndarray, variance_threshold: float = 0.95):
    """PCA-based conditioning (original method)"""
    pca = PCA(n_components=variance_threshold)
    Z_cond = pca.fit_transform(Z)
    return Z_cond, {"pca": pca, "method": "pca"}

def condition_latent_space_umap(Z: np.ndarray, n_components: int = 4):
    """UMAP-based non-linear dimensionality reduction"""
    umap_model = UMAP(
        n_components=n_components,
        n_neighbors=50,
        min_dist=0.1,
        random_state=42,
        transform_seed=42
    )
    Z_cond = umap_model.fit_transform(Z)
    return Z_cond, {"umap": umap_model, "method": "umap"}

def apply_conditioning(Z: np.ndarray, conditioning: dict):
    """Apply saved conditioning to new embeddings"""
    if conditioning["method"] == "pca":
        return conditioning["pca"].transform(Z)
    elif conditioning["method"] == "umap":
        return conditioning["umap"].transform(Z)
    else:
        raise ValueError(f"Unknown conditioning method: {conditioning['method']}")

def save_conditioning(conditioning: dict, path: str):
    with open(path, "wb") as f:
        pickle.dump(conditioning, f)

def load_conditioning(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)